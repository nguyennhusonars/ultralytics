"""Qwen3-VL class-verification batch runner.

Take a folder of crop images (organized by GT class in subfolders) and ask
Qwen3-VL to identify which class each crop belongs to. Output a structured
JSON file that the YOLO labeling tool can ingest to surface ground-truth
mistakes for hand-review.

Layout expected for IMAGE_FOLDER:

    IMAGE_FOLDER/
        bus_m/        # GT class name
            crop_000.jpg
            crop_001.jpg
        sedan/
            crop_000.jpg
        ...

Each crop's parent-folder name is its GT class label. The same name must
appear as a key in ``class_descriptions.json`` (the description is fed
into the prompt). Crops without a matching class folder are skipped.

Output: ``OUTPUT_FOLDER/verification.json`` — a list of records:

    {"filename": "bus_m/crop_000.jpg",
     "gt_class": "bus_m",
     "pred_class": "bus_l",
     "confidence": "high",
     "notes": "three axles visible, body longer than 12m"}

Notes
-----
* Greedy decoding (do_sample=False) — deterministic outputs.
* "confidence": qwen returns one of {"low", "medium", "high"}; map however
  you like on the consumer side.
* If the model can't decide (occluded, ambiguous, etc.) it returns
  ``pred_class: "unknown"``. Treat these as "no verdict" — don't auto-apply.
"""

from __future__ import annotations

import os
import re
import json
import time
import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
from qwen_vl_utils import process_vision_info


# -----------------------------
# CONFIG
# -----------------------------
IMAGE_FOLDER = "/workspace/crops"          # parent → class subfolders → images
OUTPUT_FOLDER = "/workspace/output_qwen_verify"
CLASS_DESC_FILE = "class_descriptions.json"  # next to this script by default
MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"

BATCH_SIZE = 10
NUM_WORKERS = 12
PIN_MEMORY = True

GEN_CONFIG = {
    "max_new_tokens": 160,
    "do_sample": False,        # deterministic
    "temperature": 0.0,
    "repetition_penalty": 1.05,
}


# -----------------------------
# PROMPT
# -----------------------------
def build_system_prompt(class_descriptions: dict[str, str]) -> str:
    """Render the system prompt from the class-description dict. Each entry
    is one line of pixel-evidence criteria; ordering follows insertion order
    so the user can put the most ambiguous-to-distinguish classes adjacent
    (which the model handles slightly better)."""
    class_block_lines = []
    for name, desc in class_descriptions.items():
        class_block_lines.append(f"- {name}: {desc}")
    class_block = "\n".join(class_block_lines)
    allowed_values = ", ".join(class_descriptions.keys())

    return f"""You analyze ONE cropped image of a single object.
The object was previously annotated; your job is to identify which class
it actually belongs to, based ONLY on direct pixel evidence.

Output ONLY valid JSON (double quotes). No explanation. No markdown.

GENERAL RULES
- Use ONLY pixel evidence visible in the image. No statistical inference.
- If the crop is unreadable (blur, occluded >50%, dark, off-center, mostly
  background) return pred_class = "unknown".
- "unknown" is a valid, correct answer. Prefer it over guessing.

PRIMARY OBJECT
- If the crop contains MULTIPLE objects, classify ONLY the one that is
  largest in area AND closest to the crop center.
- If two objects of different classes are similar in size and centered,
  return "unknown".

REAL VS. IMAGE OF AN OBJECT
- The object must be a REAL physical object photographed live.
- If the "object" is a printed image (billboard, poster, sticker, painting,
  bus livery showing another vehicle), a reflection (in glass, water,
  mirror), or a screen, return "unknown".
- If the object is a toy / model / scale replica, return "unknown".

LICENSE PLATE vs. VEHICLE
- If a license plate is incidental (occupies <50% of the crop area and
  the rest of the crop clearly shows a vehicle), classify the VEHICLE,
  not the plate.
- Classify as a plate class (BSD / BSV) ONLY when the plate dominates
  the crop (>50% area) AND vehicle body is barely visible.

VIEW-ANGLE FALLBACK
- For wheeled vehicles, key distinctions (axle count, body length, ride
  height, bed visibility) require a SIDE or 3/4 view to be reliable.
- If the crop is a pure rear-only or front-only view and the size/sub-type
  cannot be determined from visible features (axle hidden, length
  foreshortened), return "unknown" rather than guessing the sub-type.

ALLOWED CLASSES (pick exactly one OR "unknown")
{class_block}

OUTPUT (exact shape)
{{"pred_class":"<one of [{allowed_values}, unknown]>",
"confidence":"<low|medium|high>",
"notes":"<one short sentence citing the decisive visual evidence>"}}
""".strip()


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict | None:
    try:
        m = _JSON_RE.search(text)
        if not m:
            return None
        return json.loads(m.group())
    except Exception:
        return None


# -----------------------------
# DATASET
# -----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CropClassDataset(Dataset):
    """Walks IMAGE_FOLDER. For each image, the immediate parent folder
    name is treated as the GT class label.

    Yields (text_prompt, vision_inputs, gt_class, relative_filename) per item.
    """

    def __init__(self, image_folder: str, processor: AutoProcessor,
                 system_prompt: str, allowed_classes: set[str]):
        self.root = Path(image_folder).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"IMAGE_FOLDER not found: {self.root}")
        self.processor = processor
        self.system_prompt = system_prompt
        self.allowed_classes = allowed_classes

        files = []
        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMG_EXTS:
                continue
            # Parent folder name = GT class
            gt_class = p.parent.name
            if gt_class not in allowed_classes:
                # Skip silently; user might have extra folders that aren't
                # in their class list. The summary logs these at the end.
                continue
            files.append((p, gt_class))
        self.files = sorted(files, key=lambda x: str(x[0]))
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path, gt_class = self.files[idx]
        fname = path.relative_to(self.root).as_posix()
        image_path = str(path.resolve())

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": f"This crop was labeled as: {gt_class}. "
                                          f"Identify its actual class."},
            ]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return {
            "text": text,
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "filename": fname,
            "gt_class": gt_class,
        }


def collate_fn(batch):
    return (
        [b["text"] for b in batch],
        [b["image_inputs"] for b in batch],
        [b["video_inputs"] for b in batch],
        [b["filename"] for b in batch],
        [b["gt_class"] for b in batch],
    )


# -----------------------------
# NORMALIZE
# -----------------------------
def normalize_record(d: dict, allowed_classes: set[str]) -> dict:
    """Clamp Qwen output into the schema. Anything weird becomes 'unknown'."""
    pred = str(d.get("pred_class", "") or "").strip()
    if pred not in allowed_classes and pred != "unknown":
        pred = "unknown"
    conf = str(d.get("confidence", "") or "").strip().lower()
    if conf not in {"low", "medium", "high"}:
        conf = "low"
    notes = str(d.get("notes", "") or "").strip()
    return {"pred_class": pred, "confidence": conf, "notes": notes}


def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


# -----------------------------
# MAIN
# -----------------------------
def main():
    here = Path(__file__).resolve().parent
    desc_path = Path(CLASS_DESC_FILE)
    if not desc_path.is_absolute():
        desc_path = here / desc_path
    if not desc_path.exists():
        raise FileNotFoundError(
            f"class_descriptions.json not found at {desc_path}. "
            f"See the template alongside this script."
        )
    class_descriptions = json.loads(desc_path.read_text(encoding="utf-8"))
    if not isinstance(class_descriptions, dict) or not class_descriptions:
        raise ValueError(f"{desc_path} must be a non-empty {{name: description}} dict.")
    allowed = set(class_descriptions.keys())
    system_prompt = build_system_prompt(class_descriptions)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    out_json_path = os.path.join(OUTPUT_FOLDER, "verification.json")
    log_path = os.path.join(OUTPUT_FOLDER, "verification.log")

    print(f"🚀 Loading model: {MODEL_ID}")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    dataset = CropClassDataset(IMAGE_FOLDER, processor, system_prompt, allowed)
    if len(dataset) == 0:
        # Helpful error: probably mismatched folder names
        existing = sorted({p.parent.name for p in Path(IMAGE_FOLDER).rglob("*")
                           if p.is_file()})
        raise RuntimeError(
            f"No images found in {IMAGE_FOLDER} matching any class in "
            f"class_descriptions.json. Subfolders present: {existing}. "
            f"Allowed classes: {sorted(allowed)}"
        )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    total = len(dataset)
    print(f"📂 {total} crops in {len(allowed)} classes. "
          f"Batch={BATCH_SIZE}, Workers={NUM_WORKERS}, Device={model.device}")
    print("=" * 64)

    results: list[dict[str, Any]] = []
    start = time.time()
    done = 0
    n_match = n_disagree = n_unknown = n_parse_error = 0

    with open(log_path, "w", encoding="utf-8") as f_log, torch.inference_mode():
        for texts, image_inputs, video_inputs, filenames, gt_classes in dataloader:
            if video_inputs is not None and all(v is None for v in video_inputs):
                video_inputs = None

            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            gen_ids = model.generate(**inputs, **GEN_CONFIG)
            trimmed = [out_ids[len(in_ids):]
                       for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
            outs = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for fname, gt, out_text in zip(filenames, gt_classes, outs):
                parsed = extract_json(out_text)
                if parsed is None:
                    record = {
                        "filename": fname,
                        "gt_class": gt,
                        "pred_class": "parse_error",
                        "confidence": "low",
                        "notes": out_text[:200].replace("\n", " "),
                    }
                    icon = "❌"
                    n_parse_error += 1
                else:
                    n = normalize_record(parsed, allowed)
                    record = {"filename": fname, "gt_class": gt, **n}
                    if record["pred_class"] == "unknown":
                        icon = "🤔"; n_unknown += 1
                    elif record["pred_class"] == gt:
                        icon = "✅"; n_match += 1
                    else:
                        icon = "⚠️ "; n_disagree += 1
                results.append(record)
                line = (f"{icon} [{fname}] gt={gt} → "
                        f"pred={record['pred_class']} ({record['confidence']}) "
                        f"– {record['notes']}")
                print(line)
                f_log.write(line + "\n")

            done += len(filenames)
            elapsed = time.time() - start
            eta = (total - done) * (elapsed / max(done, 1))
            print(f"   📊 {done}/{total} ({(done/total)*100:.1f}%) | "
                  f"{elapsed/max(done,1):.2f}s/img | ETA {format_time(eta)}")
            print("-" * 64)

    summary = {
        "model": MODEL_ID,
        "image_folder": str(Path(IMAGE_FOLDER).resolve()),
        "total": total,
        "match": n_match,
        "disagree": n_disagree,
        "unknown": n_unknown,
        "parse_error": n_parse_error,
        "elapsed_seconds": int(time.time() - start),
        "results": results,
    }
    with open(out_json_path, "w", encoding="utf-8") as f_out:
        json.dump(summary, f_out, indent=2, ensure_ascii=False)
    print("=" * 64)
    print(f"🎉 Done in {format_time(summary['elapsed_seconds'])}. "
          f"match={n_match} disagree={n_disagree} unknown={n_unknown} "
          f"parse_error={n_parse_error}")
    print(f"✅ Saved: {out_json_path}")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
