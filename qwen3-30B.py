import os
import time
import json
import re
import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import torch
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
from qwen_vl_utils import process_vision_info

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_FOLDER = "/workspace/images"
OUTPUT_FOLDER = "/workspace/output_qwen30"
MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"

BATCH_SIZE = 10       # bạn có thể tăng nếu VRAM đủ
NUM_WORKERS = 16              # 20 thường nặng/overhead; 8-12 thường hợp lý
PIN_MEMORY = True

GEN_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": False,         # greedy ổn định
    "temperature": 0.0,
    "repetition_penalty": 1.05,
}

# -----------------------------
# PROMPT
# -----------------------------
SYSTEM_PROMPT = """
You analyze ONE cropped CCTV person image.

Output ONLY valid JSON (double quotes).
No explanation. No markdown.

If any attribute is unclear (blur, dark, occluded, cropped, multi-person confusion) → ["nothing"].

Use ONLY direct pixel evidence.
No statistical inference.
Evaluate each attribute independently.

TARGET

Describe ONLY the PRIMARY person (largest / centered / closest).
Ignore others.
Uncertain ownership → ["nothing"].

ALLOWED VALUES

Gender: male, female, nothing
Accessories: glasses, hat, nothing
Carrying: backpack, handbag, nothing
TypeClothing: pants, dress, nothing
UpperBodyLength: long sleeve, short sleeve, nothing
LowerBodyLength: long, shorts, nothing
Colors: black, blue, brown, gray, green, pink, purple, red, white, yellow, nothing

COLOR (Highest Priority)

Fabric only (not skin).
ONE dominant color only.
Mixed / shadowed / small / uncertain → ["nothing"].

Shade map:
navy/teal→blue
beige/khaki/tan→brown
orange/maroon→red
violet/lavender→purple
grey→gray

UPPER

long sleeve → sleeves cover most of arms
short sleeve → upper arms clearly visible
else → ["nothing"]

LOWER (Deterministic)

Outerwear hem ≠ skirt hem.

TypeClothing:

Return pants if ANY:

crotch

inner seam

two leg openings

two leg tubes

pants under outerwear

Return dress ONLY IF ALL:

single continuous skirt panel across both legs

no crotch

no two leg tubes

no pants evidence

If uncertain → pants

LowerBodyLength:

shorts → hem clearly above knee

long → otherwise

knee/hem unclear → ["nothing"]

ACCESSORIES (Strict Anti-Bias)
glasses (physical proof required)

Return glasses ONLY IF ALL:

face visible (front/side)

≥1 eye clearly visible (not dark/shadowed/blurred)

rigid eyewear overlaps eye region

visible lens OR frame rim OR bridge OR temple arm

Must rest on nose/ears.

DO NOT infer from:
hat, dark eye region, shadow, blur, reflection, style.

Eye unclear OR no clear structure OR back view → ["nothing"].

hat

Return hat ONLY if real headwear visible:
cap, hat, helmet, beanie

NOT hat:
hood, headphones, headband

Uncertain → ["nothing"]

CARRYING

backpack → worn on back + visible shoulder straps

handbag → clearly held object larger than hand with visible outline/handle

Ignore small items.
None → ["nothing"]

OUTPUT (EXACT)

{"Gender":["nothing"],
"Accessories":["nothing"],
"Carrying":["nothing"],
"TypeClothing":["nothing"],
"UpperBodyLength":["nothing"],
"UpperColor":["nothing"],
"LowerBodyLength":["nothing"],
"LowerColor":["nothing"]}
""".strip()

# -----------------------------
# MAPPING -> Your final label tokens
# -----------------------------
VALUE_MAPPING = {
    "Gender": {"male": "male", "female": "female", "nothing": "unknownGender"},
    "Accessories": {"glasses": "glasses", "hat": "hat", "nothing": None},
    "Carrying": {"backpack": "backpack", "handbag": "handbag", "nothing": None},
    "TypeClothing": {"pants": "pants", "dress": "dress", "nothing": "unknownType"},
    "UpperBodyLength": {"long sleeve": "upperLongSleeve", "short sleeve": "upperShortSleeve", "nothing": "unknownUpperSleeve"},
    "LowerBodyLength": {"long": "lowerLong", "shorts": "lowerShorts", "short": "lowerShorts", "nothing": "unknownLowerLength"},
}

ALLOWED_COLORS = {"black","blue","brown","gray","green","pink","purple","red","white","yellow"}

# -----------------------------
# DATASET (recursive)
# -----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class ImageCaptionDataset(Dataset):
    def __init__(self, image_folder: str, processor: AutoProcessor):
        self.root = Path(image_folder)
        if not self.root.exists():
            raise FileNotFoundError(f"IMAGE_FOLDER not found: {self.root}")

        self.files = sorted([p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
        self.processor = processor
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        fname = path.relative_to(self.root).as_posix()
        image_path = str(path.resolve())

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": "Extract attributes now."},
            ]},
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        return {"text": text, "image_inputs": image_inputs, "video_inputs": video_inputs, "filename": fname}

def collate_fn(batch):
    return (
        [b["text"] for b in batch],
        [b["image_inputs"] for b in batch],
        [b["video_inputs"] for b in batch],
        [b["filename"] for b in batch],
    )

# -----------------------------
# PARSE + NORMALIZE
# -----------------------------
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json(text: str):
    try:
        m = _JSON_RE.search(text)
        if not m:
            return None
        return json.loads(m.group())
    except Exception:
        return None

def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip().lower() for v in x if str(v).strip()]
    s = str(x).strip().lower()
    return [s] if s else []

def drop_nothing(vals):
    vals = as_list(vals)
    if any(v != "nothing" for v in vals):
        vals = [v for v in vals if v != "nothing"]
    return vals if vals else ["nothing"]

def first_or_nothing(vals):
    vals = drop_nothing(vals)
    return vals[0] if vals else "nothing"

def norm_color_list(vals):
    vals = drop_nothing(vals)
    if vals == ["nothing"]:
        return ["nothing"]
    out = []
    for v in vals:
        v = v.replace("grey", "gray")
        if v in ALLOWED_COLORS:
            out.append(v)
    return out if out else ["nothing"]

def normalize_record(d: dict) -> dict:
    # enforce expected keys + robust list handling
    out = {}
    out["Gender"] = first_or_nothing(d.get("Gender"))
    out["TypeClothing"] = first_or_nothing(d.get("TypeClothing"))
    out["UpperBodyLength"] = first_or_nothing(d.get("UpperBodyLength"))
    out["LowerBodyLength"] = first_or_nothing(d.get("LowerBodyLength"))

    out["Accessories"] = drop_nothing(d.get("Accessories"))
    out["Carrying"] = drop_nothing(d.get("Carrying"))
    out["UpperColor"] = norm_color_list(d.get("UpperColor"))
    out["LowerColor"] = norm_color_list(d.get("LowerColor"))

    # normalize "short" -> "shorts"
    if out["LowerBodyLength"] == "short":
        out["LowerBodyLength"] = "shorts"

    return out

# -----------------------------
# BUILD OUTPUT LABEL STRING
# -----------------------------
def build_filename_string(data: dict) -> str:
    parts = []
    parts.append(VALUE_MAPPING["Gender"].get(data["Gender"], "unknownGender"))
    parts.append(VALUE_MAPPING["TypeClothing"].get(data["TypeClothing"], "unknownType"))
    parts.append(VALUE_MAPPING["UpperBodyLength"].get(data["UpperBodyLength"], "unknownUpperSleeve"))
    parts.append(VALUE_MAPPING["LowerBodyLength"].get(data["LowerBodyLength"], "unknownLowerLength"))

    # Accessories (multi)
    acc = [VALUE_MAPPING["Accessories"].get(a) for a in data["Accessories"]]
    acc = [a for a in acc if a]
    parts.append("_".join(sorted(set(acc))) if acc else "noAccessory")

    # Carrying (multi)
    car = [VALUE_MAPPING["Carrying"].get(c) for c in data["Carrying"]]
    car = [c for c in car if c]
    parts.append("_".join(sorted(set(car))) if car else "noCarrying")

    # UpperColor / LowerColor (expect 1 dominant, but keep safe)
    up = [f"up{c}" for c in data["UpperColor"] if c in ALLOWED_COLORS]
    down = [f"down{c}" for c in data["LowerColor"] if c in ALLOWED_COLORS]
    parts.append("_".join(sorted(set(up))) if up else "upNone")
    parts.append("_".join(sorted(set(down))) if down else "downNone")

    return "_".join(parts)

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    out_path = os.path.join(OUTPUT_FOLDER, "output_class_30b.txt")

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

    dataset = ImageCaptionDataset(IMAGE_FOLDER, processor)
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
    print(f"📂 Found {total} images. Batch={BATCH_SIZE}, Workers={NUM_WORKERS}, Device={model.device}")
    print("=" * 60)

    start = time.time()
    done = 0

    with open(out_path, "w", encoding="utf-8") as f_out, torch.inference_mode():
        for texts, image_inputs, video_inputs, filenames in dataloader:
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

            # trim prompt
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
            outs = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for fname, out_text in zip(filenames, outs):
                d = extract_json(out_text)
                if d is None:
                    label = "parse_error"
                    icon = "❌"
                else:
                    norm = normalize_record(d)
                    label = build_filename_string(norm)
                    icon = "✅"
                f_out.write(f"{fname}\t{label}\n")
                print(f"{icon} [{fname}]: {label}")

            done += len(filenames)
            elapsed = time.time() - start
            eta = (total - done) * (elapsed / max(done, 1))
            print(f"   📊 {done}/{total} ({(done/total)*100:.1f}%) | {elapsed/max(done,1):.2f}s/img | ETA {format_time(eta)}")
            print("-" * 60)

    print(f"🎉 DONE! Total time: {format_time(time.time() - start)}")
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
