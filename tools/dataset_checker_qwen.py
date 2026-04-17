"""
YOLO Dataset Checker with Local Qwen VL  (v3 — fast + background)
==================================================================
Loads Qwen2-VL / Qwen2.5-VL / Qwen3-VL locally via transformers,
crops every YOLO bbox, asks the model to verify the class.

Key features over v2:
  • Background thread — browse / fix / export partial results while verification runs
  • Batch inference — multiple crops per GPU forward pass (8‒10× faster)
  • PIL‑direct — no temp-file I/O per crop
  • Shorter prompt + fewer output tokens

Usage:
    # Preload model + dataset, then open Gradio:
    python dataset_checker_qwen.py \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --yaml /workspace/datasets/data.yaml \
        --share

    # Tip: install flash-attn for 2-3x extra speed on Ampere/Ada GPUs:
    pip install flash-attn --no-build-isolation
"""

import os
import re
import json
import shutil
import time
import random
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import yaml
import numpy as np
from PIL import Image

# ── Monkey-patch gradio 4.x bug ──
try:
    import gradio_client.utils as _gc_utils
    _orig = _gc_utils._json_schema_to_python_type
    def _patched(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig(schema, defs)
    _gc_utils._json_schema_to_python_type = _patched
except Exception:
    pass

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────
@dataclass
class LabelEntry:
    image_path: str
    label_path: str
    line_idx: int
    class_id: int
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    qwen_class: Optional[str] = None
    qwen_confidence: Optional[str] = None
    qwen_reasoning: Optional[str] = None
    is_correct: Optional[bool] = None
    crop_path: Optional[str] = None


@dataclass
class DatasetInfo:
    yaml_path: str = ""
    base_dir: str = ""
    class_names: list = field(default_factory=list)
    nc: int = 0
    entries: list = field(default_factory=list)
    image_label_pairs: list = field(default_factory=list)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Default reclassification targets — can be overridden per-run via UI or config JSON
DEFAULT_RECLASS_TARGETS = {
    "sedan":     "sedan car (4-door, low profile, separate trunk)",
    "suv":       "SUV or crossover (tall body, 4-door, integrated rear)",
    "pickup":    "pickup truck (open cargo bed behind cabin)",
    "hatchback": "hatchback car (compact, rear door opens upward)",
    "truck_m":   "small-medium truck — 2 axles (4 wheels), box/flatbed body, shorter than a bus",
    "bus_m":     "van / minibus / small-medium bus — 2 axles, windows along side, carries passengers",
    "truck_l":   "large truck — 3+ axles (6+ wheels), long cargo body, NO container on top",
    "bus_l":     "large bus / tourist bus — 3+ axles, long body with many passenger windows",
    "truck_xl":  "container truck / semi-trailer — tractor head pulling a shipping container or long trailer",
    "bus_xl":    "articulated bus — two bus sections connected by a flexible joint",
}
DEFAULT_SKIP_CLASSES = {"BSD", "BSV"}
DEFAULT_ALIASES = {
    "bicycle": "bike", "cyclist": "bike",
    "motorcycle": "motorbike", "scooter": "motorbike",
    "container": "truck_xl", "semi": "truck_xl", "trailer": "truck_xl",
    "van": "bus_m", "minibus": "bus_m",
    "hatch": "hatchback",
}


# ──────────────────────────────────────────────
# Core Logic
# ──────────────────────────────────────────────
class DatasetChecker:
    def __init__(self):
        self.dataset = DatasetInfo()
        self.model = None
        self.processor = None
        self.model_id = ""
        self.results: list[LabelEntry] = []
        self.crop_dir = ""
        # Reclassification config (editable via UI / config JSON)
        self.reclass_targets: dict[str, str] = dict(DEFAULT_RECLASS_TARGETS)
        self.skip_classes: set[str] = set(DEFAULT_SKIP_CLASSES)
        self.aliases: dict[str, str] = dict(DEFAULT_ALIASES)
        self.input_classes: list[str] = []  # empty = check all (minus skip)
        # Background state
        self._running = False
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None
        self._status_text = ""
        self._total_to_check = 0
        self._autosave_path = ""

    @property
    def reclass_names(self) -> list[str]:
        return list(self.reclass_targets.keys())

    # ── Config ──
    def configure_targets(self, input_text: str, targets_text: str, skip_text: str, aliases_text: str) -> str:
        """Parse multi-line config text and update instance settings.
        input_text: comma-separated original class names to check (empty=all)
        targets_text: one per line, format 'name: description'
        skip_text: comma-separated class names to skip
        aliases_text: one per line, format 'alias: target_name'
        """
        new_targets = {}
        for line in targets_text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                name, desc = line.split(":", 1)
                new_targets[name.strip()] = desc.strip()
            else:
                new_targets[line.strip()] = ""
        if not new_targets:
            return "❌ No valid target classes found."

        self.reclass_targets = new_targets
        self.skip_classes = {c.strip() for c in skip_text.split(",") if c.strip()}
        self.input_classes = [c.strip() for c in input_text.split(",") if c.strip()]

        new_aliases = {}
        for line in aliases_text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                alias, target = line.split(":", 1)
                new_aliases[alias.strip().lower()] = target.strip()
        self.aliases = new_aliases

        return (
            f"✅ Configured {len(self.reclass_targets)} target classes, "
            f"{len(self.skip_classes)} skip classes, {len(self.aliases)} aliases.\n"
            f"Input classes: {self.input_classes if self.input_classes else '(all)'}\n"
            f"Targets: {self.reclass_names}\n"
            f"Skip: {self.skip_classes}"
        )

    def save_config(self, path: str) -> str:
        path = path.strip()
        if not path:
            path = os.path.join(self.dataset.base_dir or ".", "checker_config.json")
        cfg = {
            "input_classes": self.input_classes,
            "targets": self.reclass_targets,
            "skip_classes": sorted(self.skip_classes),
            "aliases": self.aliases,
        }
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return f"✅ Config saved to {path}"

    def load_config(self, path: str) -> tuple[str, str, str, str, str]:
        """Load config JSON. Returns (status, input_text, targets_text, skip_text, aliases_text)."""
        path = path.strip()
        if not os.path.isfile(path):
            return f"❌ File not found: {path}", "", "", "", ""
        with open(path, "r") as f:
            cfg = json.load(f)
        self.input_classes = cfg.get("input_classes", [])
        self.reclass_targets = cfg.get("targets", {})
        self.skip_classes = set(cfg.get("skip_classes", []))
        self.aliases = cfg.get("aliases", {})
        # Format back to text for UI
        input_text = ", ".join(self.input_classes)
        targets_text = "\n".join(f"{k}: {v}" for k, v in self.reclass_targets.items())
        skip_text = ", ".join(sorted(self.skip_classes))
        aliases_text = "\n".join(f"{k}: {v}" for k, v in self.aliases.items())
        status = (
            f"✅ Loaded config from {path}\n"
            f"Input: {self.input_classes if self.input_classes else '(all)'}\n"
            f"{len(self.reclass_targets)} targets, {len(self.skip_classes)} skip, {len(self.aliases)} aliases"
        )
        return status, input_text, targets_text, skip_text, aliases_text

    # ── Model ──
    def load_model(self, model_id: str, dtype: str = "bfloat16") -> str:
        import torch
        from transformers import AutoProcessor, AutoConfig

        model_id = model_id.strip()
        if not model_id:
            return "❌ Please enter a model ID."

        try:
            dtype_map = {
                "auto": "auto",
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(dtype, torch.bfloat16)

            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model_type = getattr(config, "model_type", "")
            print(f"Detected model_type: {model_type}")

            if model_type == "qwen3_vl_moe":
                from transformers import Qwen3VLMoeForConditionalGeneration as ModelClass
                model_family = "Qwen3-VL-MoE"
            elif model_type == "qwen3_vl":
                from transformers import Qwen3VLForConditionalGeneration as ModelClass
                model_family = "Qwen3-VL"
            elif model_type in ("qwen2_5_vl",):
                from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
                model_family = "Qwen2.5-VL"
            elif model_type in ("qwen2_vl",):
                from transformers import Qwen2VLForConditionalGeneration as ModelClass
                model_family = "Qwen2-VL"
            elif model_type in ("qwen3_5",):
                from transformers import Qwen3_5ForConditionalGeneration as ModelClass
                model_family = "Qwen3.5-VL"
            elif model_type in ("qwen3.5",):
                from transformers import Qwen3_5ForConditionalGeneration as ModelClass
                model_family = "Qwen3.5-VL"
            else:
                from transformers import AutoModelForVision2Seq as ModelClass
                model_family = f"Auto ({model_type})"

            print(f"Loading {model_family}: {model_id} ...")

            self.processor = AutoProcessor.from_pretrained(model_id)
            self.processor.tokenizer.padding_side = "left"
            if self.processor.tokenizer.pad_token_id is None:
                self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

            load_kwargs = dict(torch_dtype=torch_dtype, device_map="auto")
            try:
                self.model = ModelClass.from_pretrained(
                    model_id, attn_implementation="flash_attention_2", **load_kwargs
                ).eval()
                attn_info = "flash_attention_2"
            except Exception:
                print("flash_attention_2 not available, using default attention")
                print("  ➜ Install for 2-3× speedup: pip install flash-attn --no-build-isolation")
                self.model = ModelClass.from_pretrained(model_id, **load_kwargs).eval()
                attn_info = "sdpa (default)"

            self.model_id = model_id
            dev = next(self.model.parameters()).device
            mem = ""
            if dev.type == "cuda":
                gb = torch.cuda.memory_allocated(dev) / 1e9
                mem = f" | VRAM: {gb:.1f} GB"
            return f"✅ {model_family}: {model_id} on {dev} ({attn_info}, {dtype}){mem}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Failed: {e}"

    # ── Dataset ──
    def load_dataset(self, yaml_path: str, split: str = "val", max_images: int = 0) -> str:
        yaml_path = yaml_path.strip()
        if not os.path.isfile(yaml_path):
            return f"❌ File not found: {yaml_path}"

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        base_dir = os.path.dirname(os.path.abspath(yaml_path))
        class_names = data.get("names", [])
        nc = data.get("nc", len(class_names))

        self.dataset = DatasetInfo(
            yaml_path=yaml_path, base_dir=base_dir, class_names=class_names, nc=nc,
        )

        split_data = data.get(split, [])
        if isinstance(split_data, str):
            split_data = [split_data]

        pairs = []
        for img_dir_rel in split_data:
            img_dir = os.path.join(base_dir, img_dir_rel)
            if not os.path.isdir(img_dir):
                continue
            lbl_dir = img_dir.replace("/images", "/labels")
            if not os.path.isdir(lbl_dir):
                continue
            for fname in sorted(os.listdir(img_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    continue
                img_path = os.path.join(img_dir, fname)
                stem = os.path.splitext(fname)[0]
                lbl_path = os.path.join(lbl_dir, stem + ".txt")
                if os.path.isfile(lbl_path):
                    pairs.append((img_path, lbl_path))

        if max_images > 0:
            random.shuffle(pairs)
            pairs = pairs[:max_images]

        self.dataset.image_label_pairs = pairs

        entries = []
        for img_path, lbl_path in pairs:
            with open(lbl_path, "r") as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                    entries.append(LabelEntry(
                        image_path=img_path, label_path=lbl_path,
                        line_idx=line_idx, class_id=cls_id, class_name=cls_name,
                        cx=cx, cy=cy, w=w, h=h,
                    ))
        self.dataset.entries = entries

        class_counts = {}
        for e in entries:
            class_counts[e.class_name] = class_counts.get(e.class_name, 0) + 1
        stats = "\n".join(f"  {k}: {v}" for k, v in sorted(class_counts.items()))
        return (
            f"✅ Loaded: {yaml_path}\n"
            f"Split: {split} | Classes ({nc}): {class_names}\n"
            f"Images: {len(pairs)} | Objects: {len(entries)}\n"
            f"Per class:\n{stats}"
        )

    # ── Crop ──
    def crop_object(self, entry: LabelEntry, padding_ratio: float = 0.1):
        """Return cropped region as BGR numpy array."""
        img = cv2.imread(entry.image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        x_center, y_center = entry.cx * w, entry.cy * h
        box_w, box_h = entry.w * w, entry.h * h
        pad_w, pad_h = box_w * padding_ratio, box_h * padding_ratio
        x1 = max(0, int(x_center - box_w / 2 - pad_w))
        y1 = max(0, int(y_center - box_h / 2 - pad_h))
        x2 = min(w, int(x_center + box_w / 2 + pad_w))
        y2 = min(h, int(y_center + box_h / 2 + pad_h))
        crop = img[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    CROP_MAX_PX = 384  # cap longest side — fewer vision tokens ⇒ faster inference

    def crop_object_pil(self, entry: LabelEntry, padding_ratio: float = 0.1) -> Optional[Image.Image]:
        """Return cropped region as PIL RGB image (no disk I/O)."""
        crop_bgr = self.crop_object(entry, padding_ratio)
        if crop_bgr is None:
            return None
        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        # Down-scale large crops so the VL encoder sees fewer patches
        pil.thumbnail((self.CROP_MAX_PX, self.CROP_MAX_PX), Image.LANCZOS)
        return pil

    # ── Prompt ──
    def _build_prompt(self, entry: LabelEntry) -> str:
        target_lines = "\n".join(
            f"- {k}: {v}" if v else f"- {k}"
            for k, v in self.reclass_targets.items()
        )
        examples = list(self.reclass_targets.keys())[:3]
        examples_str = ', '.join(f'"{e}"' for e in examples)
        return (
            f'Reclassify this cropped traffic object into ONE of the categories below.\n'
            f'Original label: "{entry.class_name}".\n\n'
            f'CATEGORIES (pick EXACTLY one short name from the left column):\n{target_lines}\n\n'
            f'RULES:\n'
            f'- Output ONLY the short name (e.g. {examples_str}), never the description.\n'
            f'- If the object does not fit ANY category above, reply new_class="SKIP".\n\n'
            f'Reply JSON only: {{"new_class":"...","confidence":"HIGH/MEDIUM/LOW","reason":"..."}}'
        )

    def _parse_reply(self, reply: str, entry: LabelEntry) -> LabelEntry:
        try:
            # Strip <think>...</think> blocks (Qwen3.5+ thinking mode)
            clean = _THINK_RE.sub("", reply).strip()
            m = _JSON_RE.search(clean)
            if not m:
                # Fallback: try raw reply (partial think block)
                m = _JSON_RE.search(reply)
            if not m:
                entry.qwen_reasoning = f"No JSON: {reply[:300]}"
                entry.is_correct = None
                return entry
            result = json.loads(m.group())
            raw_cls = result.get("new_class", result.get("actual_class", "unknown"))
            # Normalize: strip descriptions, lowercase, fuzzy-match to valid names
            new_cls = self._normalize_class(raw_cls)
            entry.qwen_class = new_cls
            entry.is_correct = new_cls in self.reclass_names or new_cls == "SKIP"
            entry.qwen_confidence = result.get("confidence", "unknown")
            entry.qwen_reasoning = result.get("reason", "")
        except json.JSONDecodeError:
            entry.qwen_reasoning = f"Parse error: {reply[:200]}"
            entry.is_correct = None
        return entry

    def _normalize_class(self, raw: str) -> str:
        """Fuzzy-match model output to a valid reclass_names entry."""
        raw = raw.strip()
        valid = self.reclass_names
        # Exact match
        if raw in valid or raw == "SKIP":
            return raw
        # Take text before colon/comma (model sometimes appends description)
        short = raw.split(":")[0].split(",")[0].strip().lower()
        if short == "skip":
            return "SKIP"
        for name in valid:
            if name == short:
                return name
        # Partial match: check if any valid name appears in the raw text
        raw_lower = raw.lower()
        for name in valid:
            if name in raw_lower:
                return name
        # Configurable aliases
        for alias, target in self.aliases.items():
            if alias in raw_lower and target in valid:
                return target
        return raw  # Return as-is (will be marked invalid)

    # ── Batch verify (core) ──
    def _verify_batch(self, batch: list[LabelEntry]) -> list[LabelEntry]:
        """Verify a batch of entries in one GPU forward pass. Returns updated entries."""
        import torch
        from qwen_vl_utils import process_vision_info

        all_texts = []
        all_image_inputs = []
        valid = []  # (index_in_batch, entry)

        for idx, entry in enumerate(batch):
            pil_img = self.crop_object_pil(entry)
            if pil_img is None:
                entry.is_correct = None
                entry.qwen_reasoning = "Failed to crop"
                continue

            prompt = self._build_prompt(entry)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }]

            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,  # Qwen3.5: skip CoT → direct JSON
            )
            imgs, vids = process_vision_info(messages)

            all_texts.append(text_input)
            all_image_inputs.extend(imgs)
            valid.append((idx, entry))

        if not valid:
            return batch

        try:
            inputs = self.processor(
                text=all_texts,
                images=all_image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.inference_mode():
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            # Trim prompt tokens
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
            replies = self.processor.batch_decode(trimmed, skip_special_tokens=True)

            for (_, entry), reply in zip(valid, replies):
                # Debug: print first few replies
                if len(self.results) < 24:
                    print(f"  [DEBUG] {entry.class_name} → reply: {reply[:400]}")
                self._parse_reply(reply, entry)

        except Exception as e:
            # Fallback: single inference if batch fails (e.g. OOM)
            print(f"  ⚠ Batch failed ({e}), falling back to single mode.")
            for _, entry in valid:
                self._verify_single_inline(entry)

        return batch

    def _verify_single_inline(self, entry: LabelEntry) -> LabelEntry:
        """Verify one entry (fallback). No disk I/O."""
        import torch
        from qwen_vl_utils import process_vision_info

        pil_img = self.crop_object_pil(entry)
        if pil_img is None:
            entry.is_correct = None
            entry.qwen_reasoning = "Failed to crop"
            return entry

        prompt = self._build_prompt(entry)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ],
        }]

        try:
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            imgs, vids = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input], images=imgs, videos=vids,
                padding=True, return_tensors="pt",
            ).to(self.model.device)

            with torch.inference_mode():
                gen_ids = self.model.generate(
                    **inputs, max_new_tokens=128, do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
            reply = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            self._parse_reply(reply, entry)
        except Exception as e:
            entry.qwen_reasoning = f"Error: {e}"
            entry.is_correct = None
        return entry

    # ── Background verification loop ──
    def _run_verification_loop(self, entries: list[LabelEntry], batch_size: int):
        """Runs in a background thread. Appends to self.results as it goes."""
        self._running = True
        self._stop_flag = False
        total = len(entries)
        t0 = time.time()

        self.crop_dir = os.path.join(self.dataset.base_dir, "_checker_crops")
        os.makedirs(self.crop_dir, exist_ok=True)

        for start in range(0, total, batch_size):
            if self._stop_flag:
                self._status_text = f"⏹ Stopped at {len(self.results)}/{total}"
                break

            batch = entries[start:start + batch_size]
            self._verify_batch(batch)

            # Save crops for reclassified objects (new_class ≠ original)
            for i, entry in enumerate(batch):
                if entry.qwen_class and entry.qwen_class != entry.class_name:
                    crop_bgr = self.crop_object(entry)
                    if crop_bgr is not None:
                        crop_fname = f"{start+i:05d}_{entry.class_name}_to_{entry.qwen_class}_{os.path.basename(entry.image_path)}"
                        crop_path = os.path.join(self.crop_dir, crop_fname)
                        cv2.imwrite(crop_path, crop_bgr)
                        entry.crop_path = crop_path

            self.results.extend(batch)

            done = len(self.results)
            elapsed = time.time() - t0
            speed = elapsed / max(done, 1)
            eta = speed * (total - done)
            eta_h = eta / 3600
            self._status_text = (
                f"🔄 {done}/{total} ({100*done/total:.1f}%) | "
                f"{speed:.2f}s/obj | ETA {eta_h:.1f}h"
            )
            print(f"  [{done}/{total}] {speed:.2f}s/obj | ETA {eta_h:.1f}h")

            # Auto-save every 500 objects
            if done % 500 < batch_size and self._autosave_path:
                self._do_autosave()

        # Final save
        if self._autosave_path:
            self._do_autosave()

        elapsed = time.time() - t0
        if not self._stop_flag:
            self._status_text = f"✅ Done! {len(self.results)} objects in {elapsed/60:.1f}min ({elapsed/max(len(self.results),1):.2f}s/obj)"
        self._running = False

    def _do_autosave(self):
        try:
            data = [{
                "image": r.image_path, "label_file": r.label_path,
                "line_idx": r.line_idx, "class_id": r.class_id, "class_name": r.class_name,
                "bbox": [r.cx, r.cy, r.w, r.h],
                "qwen_class": r.qwen_class, "qwen_confidence": r.qwen_confidence,
                "qwen_reasoning": r.qwen_reasoning, "is_correct": r.is_correct,
            } for r in self.results]
            with open(self._autosave_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  Autosave failed: {e}")

    # ── Public API for Gradio ──
    def start_verification(self, max_objects: int = 0,
                           batch_size: int = 8) -> str:
        if self.model is None:
            return "❌ Load model first."
        if not self.dataset.entries:
            return "❌ Load dataset first."
        if self._running:
            return "⚠ Already running. Stop first or wait."

        entries = self.dataset.entries[:]
        # Use input_classes from config
        if self.input_classes:
            entries = [e for e in entries if e.class_name in self.input_classes]
        # Skip classes that don't need reclassification
        entries = [e for e in entries if e.class_name not in self.skip_classes]
        if max_objects > 0 and len(entries) > max_objects:
            random.shuffle(entries)
            entries = entries[:max_objects]

        self.results = []
        self._total_to_check = len(entries)
        self._autosave_path = os.path.join(self.dataset.base_dir, "_checker_autosave.json")
        self._status_text = f"🚀 Starting... {len(entries)} objects, batch_size={batch_size}"

        thread = threading.Thread(
            target=self._run_verification_loop,
            args=(entries, max(1, int(batch_size))),
            daemon=True,
        )
        thread.start()
        self._thread = thread
        return self._status_text

    def stop_verification(self) -> str:
        if not self._running:
            return "Not running."
        self._stop_flag = True
        return "⏹ Stop requested... will finish current batch."

    def get_live_status(self) -> str:
        if not self._status_text:
            return "Idle. Click Start to begin."
        return self._status_text

    def get_live_report(self) -> str:
        """Generate report from whatever results exist so far."""
        if not self.results:
            return "No results yet."
        return self._generate_report()

    def get_live_chart(self):
        if not self.results:
            return None
        return self._generate_summary_image()

    def get_live_gallery(self):
        return self._get_mismatch_gallery()

    # ── Report ──
    def _generate_report(self) -> str:
        # Filter out SKIP results (standalone pedestrians) — they don't need reclassification
        active = [r for r in self.results if r.qwen_class != "SKIP"]
        skipped = len(self.results) - len(active)
        total = len(active)
        valid = sum(1 for r in active if r.is_correct is True)
        invalid = sum(1 for r in active if r.is_correct is False)
        unknown = total - valid - invalid

        lines = [
            "=" * 60,
            "  RECLASSIFICATION REPORT",
            "=" * 60,
            f"Model: {self.model_id}",
            f"Status: {'🔄 Running...' if self._running else '✅ Complete'}",
            f"Total checked: {len(self.results)}" + (f" / {self._total_to_check}" if self._running else ""),
            f"  → Reclassified: {total} | Skipped (pedestrian): {skipped}",
            f"✅ Valid:    {valid} ({100*valid/max(total,1):.1f}%)",
            f"⚠ Invalid:  {invalid} ({100*invalid/max(total,1):.1f}%)",
            f"❓ Error:    {unknown} ({100*unknown/max(total,1):.1f}%)",
            "",
        ]

        # Reclassification mapping: old_class → {new_class: count} (exclude SKIP)
        reclass_map = {}
        for r in active:
            if r.qwen_class:
                m = reclass_map.setdefault(r.class_name, {})
                m[r.qwen_class] = m.get(r.qwen_class, 0) + 1

        lines.append("RECLASSIFICATION MAPPING:")
        lines.append("-" * 55)
        for old_cls in sorted(reclass_map.keys()):
            lines.append(f"\n  {old_cls} ({sum(reclass_map[old_cls].values())} objects):")
            for new_cls, cnt in sorted(reclass_map[old_cls].items(), key=lambda x: -x[1]):
                pct = 100 * cnt / sum(reclass_map[old_cls].values())
                marker = "✅" if new_cls in self.reclass_names else ("⏭" if new_cls == "SKIP" else "⚠")
                lines.append(f"    {marker} → {new_cls:<15} {cnt:>6} ({pct:5.1f}%)")

        # New class distribution (exclude SKIP)
        new_dist = {}
        for r in active:
            if r.qwen_class and r.is_correct is True:
                new_dist[r.qwen_class] = new_dist.get(r.qwen_class, 0) + 1

        lines += ["", "NEW CLASS DISTRIBUTION:"]
        lines.append(f"{'New Class':<15} {'Count':>8} {'%':>7}")
        lines.append("-" * 35)
        for cls in self.reclass_names:
            cnt = new_dist.get(cls, 0)
            if cnt > 0:
                lines.append(f"{cls:<15} {cnt:>8} {100*cnt/max(valid,1):>6.1f}%")

        # Confidence breakdown (exclude SKIP)
        conf_counts = {}
        for r in active:
            if r.qwen_confidence:
                conf_counts[r.qwen_confidence] = conf_counts.get(r.qwen_confidence, 0) + 1
        lines += ["", "CONFIDENCE:"]
        for c in ["HIGH", "MEDIUM", "LOW"]:
            cnt = conf_counts.get(c, 0)
            lines.append(f"  {c}: {cnt} ({100*cnt/max(total,1):.1f}%)")

        return "\n".join(lines)

    def _generate_summary_image(self):
        # --- Chart 1: New class distribution bar chart (exclude SKIP) ---
        active = [r for r in self.results if r.qwen_class != "SKIP"]
        new_dist = {}
        for r in active:
            if r.qwen_class and r.is_correct is True:
                new_dist[r.qwen_class] = new_dist.get(r.qwen_class, 0) + 1

        new_classes = [c for c in self.reclass_names if new_dist.get(c, 0) > 0]
        if not new_classes:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        counts = [new_dist.get(c, 0) for c in new_classes]
        colors = plt.cm.Set3(np.linspace(0, 1, len(new_classes)))
        bars = axes[0].barh(new_classes, counts, color=colors)
        axes[0].set_xlabel("Count")
        axes[0].set_title("New Class Distribution")
        for bar, cnt in zip(bars, counts):
            axes[0].text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                         str(cnt), va="center", fontsize=9)
        axes[0].invert_yaxis()

        # --- Chart 2: Reclassification heatmap (old → new, exclude SKIP) ---
        old_classes = sorted(set(r.class_name for r in active))
        matrix = np.zeros((len(old_classes), len(new_classes)), dtype=int)
        o2i = {c: i for i, c in enumerate(old_classes)}
        n2i = {c: i for i, c in enumerate(new_classes)}
        for r in active:
            if r.qwen_class in n2i and r.class_name in o2i:
                matrix[o2i[r.class_name]][n2i[r.qwen_class]] += 1

        im = axes[1].imshow(matrix, cmap="YlOrRd", aspect="auto")
        axes[1].set_xticks(range(len(new_classes)))
        axes[1].set_yticks(range(len(old_classes)))
        axes[1].set_xticklabels(new_classes, rotation=45, ha="right", fontsize=8)
        axes[1].set_yticklabels(old_classes, fontsize=9)
        axes[1].set_xlabel("New Class (Qwen)"); axes[1].set_ylabel("Original Label")
        axes[1].set_title("Reclassification Heatmap")
        for i in range(len(old_classes)):
            for j in range(len(new_classes)):
                if matrix[i][j] > 0:
                    axes[1].text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=axes[1])

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img.copy()

    def _get_mismatch_gallery(self):
        """Show reclassified objects (exclude SKIP)."""
        gallery = []
        for r in self.results:
            if r.crop_path and r.qwen_class and r.qwen_class != "SKIP" and r.qwen_class != r.class_name:
                gallery.append((r.crop_path, f"{r.class_name}→{r.qwen_class} ({r.qwen_confidence})"))
        return gallery[:100] if gallery else []

    def get_results_table(self):
        rows = []
        for i, r in enumerate(self.results):
            # Skip pedestrians from the table
            if r.qwen_class == "SKIP":
                continue
            if r.is_correct is None:
                status = "❓"
            elif r.qwen_class and r.qwen_class != r.class_name:
                status = "🔀"  # reclassified
            else:
                status = "✅"  # same class
            rows.append([
                i, status, os.path.basename(r.image_path), r.line_idx,
                r.class_name, r.qwen_class or "N/A",
                r.qwen_confidence or "N/A", r.qwen_reasoning or "",
            ])
        return rows

    def preview_entry(self, result_idx: int):
        result_idx = int(result_idx)
        if result_idx < 0 or result_idx >= len(self.results):
            return None, None, "Invalid index (results so far: {})".format(len(self.results))
        r = self.results[result_idx]
        img = cv2.imread(r.image_path)
        if img is None:
            return None, None, "Cannot read image"
        h, w = img.shape[:2]
        x1 = int((r.cx - r.w / 2) * w); y1 = int((r.cy - r.h / 2) * h)
        x2 = int((r.cx + r.w / 2) * w); y2 = int((r.cy + r.h / 2) * h)
        color = (0, 165, 255) if (r.qwen_class and r.qwen_class != r.class_name) else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        lbl = f"Label:{r.class_name}" + (f"|Qwen:{r.qwen_class}" if r.qwen_class else "")
        cv2.putText(img, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop = self.crop_object(r)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop is not None else None
        info = (
            f"Image: {os.path.basename(r.image_path)}\n"
            f"Line: {r.line_idx} | Label: {r.class_name} (id={r.class_id})\n"
            f"Qwen: {r.qwen_class} | Conf: {r.qwen_confidence}\n"
            f"Reason: {r.qwen_reasoning}"
        )
        return img_rgb, crop_rgb, info

    # ── Fix ──
    def fix_label(self, result_idx: int, new_class: str, action: str) -> str:
        result_idx = int(result_idx)
        if result_idx < 0 or result_idx >= len(self.results):
            return "Invalid index"
        r = self.results[result_idx]
        with open(r.label_path, "r") as f:
            lines = f.readlines()

        if action == "delete":
            if r.line_idx < len(lines):
                shutil.copy2(r.label_path, r.label_path + ".bak")
                del lines[r.line_idx]
                with open(r.label_path, "w") as f:
                    f.writelines(lines)
                return f"✅ Deleted line {r.line_idx} from {os.path.basename(r.label_path)}"
            return "❌ Line out of range"

        elif action == "relabel":
            if new_class not in self.dataset.class_names:
                return f"❌ Unknown class: {new_class}. Valid: {self.dataset.class_names}"
            new_id = self.dataset.class_names.index(new_class)
            if r.line_idx < len(lines):
                shutil.copy2(r.label_path, r.label_path + ".bak")
                parts = lines[r.line_idx].strip().split()
                parts[0] = str(new_id)
                lines[r.line_idx] = " ".join(parts) + "\n"
                with open(r.label_path, "w") as f:
                    f.writelines(lines)
                r.class_id = new_id
                r.class_name = new_class
                return f"✅ Relabeled to '{new_class}' in {os.path.basename(r.label_path)}"
            return "❌ Line out of range"
        return "❌ Unknown action"

    def batch_fix(self, apply_suggestions: bool, min_confidence: str = "HIGH") -> str:
        """Batch-apply reclassifications using reclass_names as the new class list."""
        if not self.results:
            return "No results yet."
        # Reclassified = valid new class that differs from original
        reclassified = [r for r in self.results
                        if r.is_correct is True and r.qwen_class and r.qwen_class != r.class_name]
        if not reclassified:
            return "No reclassifications to apply."

        conf_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        min_val = conf_order.get(min_confidence, 0)
        new_names = self.reclass_names
        fixed, skipped, errors = 0, 0, []

        for r in reclassified:
            if conf_order.get(r.qwen_confidence, 0) < min_val:
                skipped += 1
                continue
            if apply_suggestions and r.qwen_class in new_names:
                new_id = new_names.index(r.qwen_class)
                try:
                    with open(r.label_path, "r") as f:
                        lines = f.readlines()
                    if r.line_idx < len(lines):
                        if not os.path.exists(r.label_path + ".bak"):
                            shutil.copy2(r.label_path, r.label_path + ".bak")
                        parts = lines[r.line_idx].strip().split()
                        parts[0] = str(new_id)
                        lines[r.line_idx] = " ".join(parts) + "\n"
                        with open(r.label_path, "w") as f:
                            f.writelines(lines)
                        r.class_id = new_id
                        r.class_name = r.qwen_class
                        fixed += 1
                except Exception as e:
                    errors.append(str(e))

        msg = f"Fixed: {fixed} | Skipped (low conf): {skipped} | Errors: {len(errors)}"
        if fixed > 0:
            msg += f"\n\n⚠️ Label files now use target class indices (0-{len(new_names)-1})."
            msg += f"\nUpdate data.yaml 'names' to: {new_names}"
        return msg

    def export_results(self, output_path: str) -> str:
        if not self.results:
            return "No results yet."
        data = [{
            "image": r.image_path, "label_file": r.label_path,
            "line_idx": r.line_idx, "class_id": r.class_id,
            "original_class": r.class_name,
            "new_class": r.qwen_class,
            "bbox": [r.cx, r.cy, r.w, r.h],
            "confidence": r.qwen_confidence,
            "reason": r.qwen_reasoning,
            "valid": r.is_correct,
        } for r in self.results]
        output_path = output_path.strip() or os.path.join(self.dataset.base_dir, "verification_results.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return f"✅ Exported {len(data)} results to {output_path}"


# ──────────────────────────────────────────────
# Gradio App
# ──────────────────────────────────────────────
def build_app(preloaded_checker=None):
    checker = preloaded_checker or DatasetChecker()

    with gr.Blocks(title="YOLO Dataset Checker (Local Qwen VL)", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔍 YOLO Dataset Checker — Local Qwen VL")
        gr.Markdown(
            "Verify YOLO labels by cropping objects and asking a Qwen VL model.\n"
            "Verification runs **in the background** — browse / fix / export results while it's running."
        )

        # ──── Tab 1: Setup ────
        with gr.Tab("1. Setup"):
            gr.Markdown("### Load Model")
            gr.Markdown(
                "Auto-detects model class from config.json.\n\n"
                "| Model | VRAM | ID |\n"
                "|-------|------|----|\n"
                "| Qwen2-VL-2B | ~5 GB | `Qwen/Qwen2-VL-2B-Instruct` |\n"
                "| Qwen2.5-VL-3B | ~7 GB | `Qwen/Qwen2.5-VL-3B-Instruct` |\n"
                "| Qwen2.5-VL-7B | ~16 GB | `Qwen/Qwen2.5-VL-7B-Instruct` |\n"
                "| Qwen3-VL-8B | ~18 GB | `Qwen/Qwen3-VL-8B-Instruct` |\n"
                "| Qwen3-VL-30B-A3B | ~20 GB | `Qwen/Qwen3-VL-30B-A3B-Instruct` |\n"
            )
            with gr.Row():
                model_id_input = gr.Textbox(
                    label="Model ID (HuggingFace)",
                    value="Qwen/Qwen2-VL-2B-Instruct",
                )
                dtype_select = gr.Dropdown(
                    label="Dtype", choices=["bfloat16", "float16", "auto"], value="bfloat16"
                )
            load_model_btn = gr.Button("🚀 Load Model", variant="primary")
            preload_status = f"✅ Model preloaded: {checker.model_id}" if checker.model else ""
            model_status = gr.Textbox(label="Model Status", interactive=False, value=preload_status)

            gr.Markdown("---")
            gr.Markdown("### Load Dataset")
            with gr.Row():
                yaml_input = gr.Textbox(label="data.yaml Path", value="/workspace/dataset/data.yaml")
                split_select = gr.Dropdown(label="Split", choices=["train", "val", "test"], value="val")
                max_images = gr.Number(label="Max Images (0=all)", value=0, precision=0)
            load_data_btn = gr.Button("Load Dataset", variant="primary")
            preload_ds = ""
            if checker.dataset.yaml_path:
                preload_ds = f"✅ Preloaded: {len(checker.dataset.entries)} objects from {checker.dataset.yaml_path}"
            data_status = gr.Textbox(label="Dataset Info", interactive=False, lines=8, value=preload_ds)

            def _load_model_wrapper(mid, dt):
                yield "⏳ Loading model... (this may take a few minutes)"
                result = checker.load_model(mid, dt)
                yield result

            load_model_btn.click(_load_model_wrapper, inputs=[model_id_input, dtype_select], outputs=model_status)
            load_data_btn.click(checker.load_dataset, inputs=[yaml_input, split_select, max_images], outputs=data_status)

            gr.Markdown("---")
            gr.Markdown(
                "### Reclassification Config\n"
                "**Input classes**: which original label classes to reclassify (comma-sep, empty=all).\n"
                "**Target classes**: output categories the model picks from (one per line: `name: description`).\n"
                "**Skip classes**: never sent for reclassification.\n"
                "**Aliases**: fuzzy-match model output (one per line: `alias: target_name`)."
            )
            default_input_text = ", ".join(checker.input_classes) if checker.input_classes else ""
            default_targets_text = "\n".join(
                f"{k}: {v}" for k, v in checker.reclass_targets.items()
            )
            default_skip_text = ", ".join(sorted(checker.skip_classes))
            default_aliases_text = "\n".join(
                f"{k}: {v}" for k, v in checker.aliases.items()
            )
            input_classes_input = gr.Textbox(
                label="Input Classes to check (comma-sep, empty=all)",
                value=default_input_text,
                placeholder="e.g. motorbike, motorbike_standalone, motorbike_with_driver",
            )
            targets_input = gr.Textbox(
                label="Target Output Classes (name: description, one per line)",
                value=default_targets_text, lines=10, max_lines=30,
            )
            with gr.Row():
                skip_input = gr.Textbox(
                    label="Skip Classes (comma-sep)",
                    value=default_skip_text,
                )
                aliases_input = gr.Textbox(
                    label="Aliases (alias: target, one per line)",
                    value=default_aliases_text, lines=4, max_lines=15,
                )
            apply_config_btn = gr.Button("✅ Apply Config", variant="primary")
            config_status = gr.Textbox(label="Config Status", interactive=False)
            apply_config_btn.click(
                checker.configure_targets,
                inputs=[input_classes_input, targets_input, skip_input, aliases_input],
                outputs=[config_status],
            )

            gr.Markdown("#### Save / Load Config")
            with gr.Row():
                config_path_input = gr.Textbox(
                    label="Config JSON Path",
                    placeholder="checker_config.json",
                )
                save_config_btn = gr.Button("💾 Save")
                load_config_btn = gr.Button("📂 Load")
            save_config_btn.click(
                checker.save_config, inputs=[config_path_input], outputs=[config_status],
            )
            load_config_btn.click(
                checker.load_config,
                inputs=[config_path_input],
                outputs=[config_status, input_classes_input, targets_input, skip_input, aliases_input],
            )

        # ──── Tab 2: Verify ────
        with gr.Tab("2. Verify"):
            gr.Markdown(
                "Verification runs **in the background**. "
                "Switch to Browse/Fix/Export tabs anytime — partial results are always available."
            )
            with gr.Row():
                max_objects = gr.Number(label="Max Objects (0=all)", value=0, precision=0)
                batch_size_input = gr.Number(label="Batch Size", value=8, precision=0)
            gr.Markdown("*Input/target classes are configured in the Setup tab.*")
            with gr.Row():
                start_btn = gr.Button("🚀 Start Verification", variant="primary")
                stop_btn = gr.Button("⏹ Stop", variant="stop")
                refresh_status_btn = gr.Button("🔄 Refresh Status")
            status_text = gr.Textbox(label="Live Status", interactive=False, lines=2)

            gr.Markdown("### Live Report (click Refresh to update)")
            refresh_report_btn = gr.Button("🔄 Refresh Report & Chart")
            report_text = gr.Textbox(label="Report", interactive=False, lines=25)
            summary_chart = gr.Image(label="Summary Chart", type="numpy")
            mismatch_gallery = gr.Gallery(label="🔀 Reclassified (old→new)", columns=5, height=400)

            start_btn.click(
                checker.start_verification,
                inputs=[max_objects, batch_size_input],
                outputs=[status_text],
            )
            stop_btn.click(checker.stop_verification, outputs=[status_text])
            refresh_status_btn.click(checker.get_live_status, outputs=[status_text])
            refresh_report_btn.click(
                lambda: (checker.get_live_report(), checker.get_live_chart(), checker.get_live_gallery()),
                outputs=[report_text, summary_chart, mismatch_gallery],
            )

        # ──── Tab 3: Browse ────
        with gr.Tab("3. Browse"):
            gr.Markdown("Browse results collected so far (works while verification is running).")
            with gr.Row():
                result_filter = gr.Radio(
                    label="Filter", choices=["All", "Reclassified", "Same", "Error"], value="All"
                )
                refresh_btn = gr.Button("🔄 Refresh Table")
            results_table = gr.Dataframe(
                headers=["Idx", "Status", "Image", "Line", "Original", "New Class", "Conf", "Reason"],
                interactive=False, wrap=True,
            )

            def get_filtered(filt):
                rows = checker.get_results_table()
                if filt == "Reclassified": rows = [r for r in rows if r[1] == "🔀"]
                elif filt == "Same": rows = [r for r in rows if r[1] == "✅"]
                elif filt == "Error": rows = [r for r in rows if r[1] == "❓"]
                return rows

            refresh_btn.click(get_filtered, inputs=[result_filter], outputs=[results_table])

            gr.Markdown("---")
            with gr.Row():
                preview_idx = gr.Number(label="Result Index", value=0, precision=0)
                preview_btn = gr.Button("Preview")
            with gr.Row():
                preview_img = gr.Image(label="Image + BBox", type="numpy")
                preview_crop = gr.Image(label="Crop", type="numpy")
            preview_info = gr.Textbox(label="Details", interactive=False, lines=4)
            preview_btn.click(checker.preview_entry, inputs=[preview_idx],
                              outputs=[preview_img, preview_crop, preview_info])

        # ──── Tab 4: Fix ────
        with gr.Tab("4. Fix"):
            gr.Markdown("### Single Fix")
            with gr.Row():
                fix_idx = gr.Number(label="Result Index", value=0, precision=0)
                fix_action = gr.Radio(label="Action", choices=["relabel", "delete"], value="relabel")
                fix_class = gr.Textbox(label="New Class", placeholder="car")
            fix_btn = gr.Button("Apply Fix", variant="primary")
            fix_status = gr.Textbox(label="Status", interactive=False)
            fix_btn.click(lambda i, c, a: checker.fix_label(int(i), c, a),
                          inputs=[fix_idx, fix_class, fix_action], outputs=[fix_status])

            gr.Markdown("---")
            gr.Markdown("### Batch Fix")
            with gr.Row():
                batch_apply = gr.Radio(label="Apply Qwen suggestions", choices=["Yes", "No"], value="Yes")
                batch_conf = gr.Dropdown(label="Min Confidence", choices=["HIGH", "MEDIUM", "LOW"], value="HIGH")
            batch_btn = gr.Button("🔧 Batch Fix", variant="stop")
            batch_status = gr.Textbox(label="Status", interactive=False)
            batch_btn.click(lambda a, c: checker.batch_fix(a == "Yes", c),
                            inputs=[batch_apply, batch_conf], outputs=[batch_status])

        # ──── Tab 5: Export ────
        with gr.Tab("5. Export"):
            export_path = gr.Textbox(label="Output Path (empty=auto)", placeholder="results.json")
            export_btn = gr.Button("Export JSON", variant="primary")
            export_status = gr.Textbox(label="Status", interactive=False)
            export_btn.click(checker.export_results, inputs=[export_path], outputs=[export_status])

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLO Dataset Checker with Local Qwen VL")
    parser.add_argument("--model", type=str, default="", help="Model ID to preload")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "auto"])
    parser.add_argument("--yaml", type=str, default="", help="data.yaml path to preload")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    _preloaded_checker = None
    if args.model:
        _preloaded_checker = DatasetChecker()
        print(_preloaded_checker.load_model(args.model, args.dtype))
        if args.yaml:
            print(_preloaded_checker.load_dataset(args.yaml, args.split))

    app = build_app(preloaded_checker=_preloaded_checker)
    app.queue(default_concurrency_limit=5)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
