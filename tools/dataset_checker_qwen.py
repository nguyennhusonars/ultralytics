"""
YOLO Dataset Checker with Qwen Vision Model
=============================================
Scans YOLO dataset (images + labels), crops each object by bounding box,
feeds to Qwen VL model to verify class correctness.
Provides analysis, visualization, and label fixing capability.

Usage:
    python dataset_checker_qwen.py

Supports Qwen via OpenAI-compatible API (Dashscope, vLLM, Ollama, etc.)
"""

import os
import sys
import json
import base64
import shutil
import time
import random
import traceback
from pathlib import Path
from io import BytesIO
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import yaml
import numpy as np

# ── Monkey-patch gradio 4.38.1 bug: additionalProperties=True crashes schema parsing ──
import gradio_client.utils as _gc_utils

_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type

def _patched_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)

_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
# ── End monkey-patch ──

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────
@dataclass
class LabelEntry:
    image_path: str
    label_path: str
    line_idx: int  # line number in label file
    class_id: int
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    # Verification results
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
    entries: list = field(default_factory=list)  # list of LabelEntry
    image_label_pairs: list = field(default_factory=list)  # [(img_path, lbl_path), ...]


# ──────────────────────────────────────────────
# Core Logic
# ──────────────────────────────────────────────
class DatasetChecker:
    def __init__(self):
        self.dataset = DatasetInfo()
        self.client: Optional[OpenAI] = None
        self.model_name = "qwen-vl-plus"
        self.results: list[LabelEntry] = []
        self.crop_dir = ""

    def connect_api(self, api_key: str, base_url: str, model_name: str) -> str:
        """Connect to Qwen API (OpenAI-compatible)."""
        if not api_key or not base_url:
            return "❌ Please provide API key and base URL."
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name or "qwen-vl-plus"
        # Quick test
        try:
            self.client.models.list()
            return f"✅ Connected to {base_url} | Model: {self.model_name}"
        except Exception as e:
            # Connection might still work for chat, just can't list models
            return f"⚠️ Connected to {base_url} (model list unavailable: {e}). Will try on first verification."

    def load_dataset(self, yaml_path: str, split: str = "val", max_images: int = 0) -> str:
        """Load YOLO dataset from YAML file."""
        yaml_path = yaml_path.strip()
        if not os.path.isfile(yaml_path):
            return f"❌ File not found: {yaml_path}"

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        base_dir = os.path.dirname(os.path.abspath(yaml_path))
        class_names = data.get("names", [])
        nc = data.get("nc", len(class_names))

        self.dataset = DatasetInfo(
            yaml_path=yaml_path,
            base_dir=base_dir,
            class_names=class_names,
            nc=nc,
        )

        # Resolve image directories
        split_data = data.get(split, [])
        if isinstance(split_data, str):
            split_data = [split_data]

        pairs = []
        for img_dir_rel in split_data:
            img_dir = os.path.join(base_dir, img_dir_rel)
            if not os.path.isdir(img_dir):
                continue
            # Labels dir: replace /images with /labels
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

        # Parse all labels
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
                        image_path=img_path,
                        label_path=lbl_path,
                        line_idx=line_idx,
                        class_id=cls_id,
                        class_name=cls_name,
                        cx=cx, cy=cy, w=w, h=h,
                    ))

        self.dataset.entries = entries

        # Stats summary
        class_counts = {}
        for e in entries:
            class_counts[e.class_name] = class_counts.get(e.class_name, 0) + 1

        stats = "\n".join(f"  {k}: {v}" for k, v in sorted(class_counts.items()))
        return (
            f"✅ Loaded dataset: {yaml_path}\n"
            f"Split: {split}\n"
            f"Classes ({nc}): {class_names}\n"
            f"Images: {len(pairs)}\n"
            f"Total objects: {len(entries)}\n"
            f"Per class:\n{stats}"
        )

    def crop_object(self, entry: LabelEntry, padding_ratio: float = 0.1) -> np.ndarray:
        """Crop object from image using YOLO bbox with padding."""
        img = cv2.imread(entry.image_path)
        if img is None:
            return None
        h, w = img.shape[:2]

        # Convert YOLO format to pixel coords
        x_center = entry.cx * w
        y_center = entry.cy * h
        box_w = entry.w * w
        box_h = entry.h * h

        # Add padding
        pad_w = box_w * padding_ratio
        pad_h = box_h * padding_ratio

        x1 = max(0, int(x_center - box_w / 2 - pad_w))
        y1 = max(0, int(y_center - box_h / 2 - pad_h))
        x2 = min(w, int(x_center + box_w / 2 + pad_w))
        y2 = min(h, int(y_center + box_h / 2 + pad_h))

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def image_to_base64(self, img_np: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        # Resize if too large (max 512px on longest side for efficiency)
        max_side = 512
        w, h = pil_img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def verify_single(self, entry: LabelEntry) -> LabelEntry:
        """Verify a single crop with Qwen VL model."""
        crop = self.crop_object(entry)
        if crop is None:
            entry.is_correct = None
            entry.qwen_reasoning = "Failed to crop image"
            return entry

        b64 = self.image_to_base64(crop)
        class_list = ", ".join(self.dataset.class_names)

        prompt = (
            f"You are verifying labels for a vehicle detection dataset.\n"
            f"The possible classes are: [{class_list}].\n"
            f"This cropped image is labeled as: \"{entry.class_name}\".\n\n"
            f"Look at this image carefully and answer:\n"
            f"1. What object is actually in this image? Pick from the class list above.\n"
            f"2. Is the label \"{entry.class_name}\" correct? Answer YES or NO.\n"
            f"3. Your confidence: HIGH, MEDIUM, or LOW.\n"
            f"4. Brief reason.\n\n"
            f"Reply in this exact JSON format:\n"
            f'{{"actual_class": "...", "label_correct": true/false, "confidence": "HIGH/MEDIUM/LOW", "reason": "..."}}'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                max_tokens=300,
                temperature=0.1,
            )
            reply = response.choices[0].message.content.strip()

            # Parse JSON from response
            # Handle case where model wraps in markdown code block
            if "```json" in reply:
                reply = reply.split("```json")[1].split("```")[0].strip()
            elif "```" in reply:
                reply = reply.split("```")[1].split("```")[0].strip()

            result = json.loads(reply)
            entry.qwen_class = result.get("actual_class", "unknown")
            entry.is_correct = result.get("label_correct", None)
            entry.qwen_confidence = result.get("confidence", "unknown")
            entry.qwen_reasoning = result.get("reason", "")

        except json.JSONDecodeError:
            # Try to extract info from non-JSON response
            entry.qwen_reasoning = f"Raw response: {reply[:200]}"
            entry.is_correct = None
        except Exception as e:
            entry.qwen_reasoning = f"API error: {str(e)}"
            entry.is_correct = None

        return entry

    def run_verification(self, max_objects: int = 100, classes_to_check: list = None,
                         progress=gr.Progress()) -> tuple:
        """Run verification on dataset entries."""
        if not self.client:
            return "❌ Connect to API first.", None, None

        if not self.dataset.entries:
            return "❌ Load dataset first.", None, None

        entries = self.dataset.entries[:]
        if classes_to_check:
            entries = [e for e in entries if e.class_name in classes_to_check]

        if max_objects > 0 and len(entries) > max_objects:
            random.shuffle(entries)
            entries = entries[:max_objects]

        # Create crop directory
        self.crop_dir = os.path.join(self.dataset.base_dir, "_checker_crops")
        os.makedirs(self.crop_dir, exist_ok=True)

        self.results = []
        total = len(entries)

        for i, entry in enumerate(entries):
            progress((i + 1) / total, desc=f"Checking {i+1}/{total}: {entry.class_name}")
            result = self.verify_single(entry)

            # Save crop for visualization
            crop = self.crop_object(entry)
            if crop is not None:
                crop_fname = f"{i:05d}_{entry.class_name}_{os.path.basename(entry.image_path)}"
                crop_path = os.path.join(self.crop_dir, crop_fname)
                cv2.imwrite(crop_path, crop)
                result.crop_path = crop_path

            self.results.append(result)
            # Rate limiting
            time.sleep(0.3)

        # Generate report
        report = self._generate_report()
        summary_img = self._generate_summary_image()
        mismatch_gallery = self._get_mismatch_gallery()

        return report, summary_img, mismatch_gallery

    def _generate_report(self) -> str:
        """Generate text report of verification results."""
        total = len(self.results)
        correct = sum(1 for r in self.results if r.is_correct is True)
        incorrect = sum(1 for r in self.results if r.is_correct is False)
        unknown = sum(1 for r in self.results if r.is_correct is None)

        lines = [
            "=" * 60,
            "  DATASET VERIFICATION REPORT",
            "=" * 60,
            f"Total objects checked: {total}",
            f"✅ Correct labels:     {correct} ({100*correct/max(total,1):.1f}%)",
            f"❌ Incorrect labels:   {incorrect} ({100*incorrect/max(total,1):.1f}%)",
            f"❓ Unknown/errors:     {unknown} ({100*unknown/max(total,1):.1f}%)",
            "",
        ]

        # Per-class breakdown
        class_stats = {}
        for r in self.results:
            if r.class_name not in class_stats:
                class_stats[r.class_name] = {"total": 0, "correct": 0, "incorrect": 0, "unknown": 0}
            class_stats[r.class_name]["total"] += 1
            if r.is_correct is True:
                class_stats[r.class_name]["correct"] += 1
            elif r.is_correct is False:
                class_stats[r.class_name]["incorrect"] += 1
            else:
                class_stats[r.class_name]["unknown"] += 1

        lines.append("PER-CLASS BREAKDOWN:")
        lines.append("-" * 60)
        lines.append(f"{'Class':<15} {'Total':>6} {'Correct':>8} {'Wrong':>7} {'Accuracy':>9}")
        lines.append("-" * 60)
        for cls, st in sorted(class_stats.items()):
            acc = 100 * st["correct"] / max(st["total"] - st["unknown"], 1)
            lines.append(f"{cls:<15} {st['total']:>6} {st['correct']:>8} {st['incorrect']:>7} {acc:>8.1f}%")

        # Confusion-like summary
        if incorrect > 0:
            lines.append("")
            lines.append("MISLABEL PATTERNS (labeled → Qwen says):")
            lines.append("-" * 60)
            patterns = {}
            for r in self.results:
                if r.is_correct is False and r.qwen_class:
                    key = f"{r.class_name} → {r.qwen_class}"
                    patterns[key] = patterns.get(key, 0) + 1
            for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
                lines.append(f"  {pattern}: {count}")

        # Sample mismatches
        mismatches = [r for r in self.results if r.is_correct is False]
        if mismatches:
            lines.append("")
            lines.append("SAMPLE MISMATCHES:")
            lines.append("-" * 60)
            for r in mismatches[:20]:
                img_name = os.path.basename(r.image_path)
                lines.append(
                    f"  {img_name} line {r.line_idx}: "
                    f"labeled={r.class_name}, qwen={r.qwen_class} "
                    f"({r.qwen_confidence}) - {r.qwen_reasoning}"
                )

        return "\n".join(lines)

    def _generate_summary_image(self) -> Optional[np.ndarray]:
        """Generate a visual summary chart."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Per-class accuracy bar chart
            class_stats = {}
            for r in self.results:
                if r.class_name not in class_stats:
                    class_stats[r.class_name] = {"correct": 0, "incorrect": 0, "unknown": 0}
                if r.is_correct is True:
                    class_stats[r.class_name]["correct"] += 1
                elif r.is_correct is False:
                    class_stats[r.class_name]["incorrect"] += 1
                else:
                    class_stats[r.class_name]["unknown"] += 1

            classes = sorted(class_stats.keys())
            correct_vals = [class_stats[c]["correct"] for c in classes]
            incorrect_vals = [class_stats[c]["incorrect"] for c in classes]
            unknown_vals = [class_stats[c]["unknown"] for c in classes]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Stacked bar chart
            x = np.arange(len(classes))
            axes[0].bar(x, correct_vals, label="Correct", color="#4CAF50")
            axes[0].bar(x, incorrect_vals, bottom=correct_vals, label="Incorrect", color="#F44336")
            axes[0].bar(x, unknown_vals,
                        bottom=[c + i for c, i in zip(correct_vals, incorrect_vals)],
                        label="Unknown", color="#FFC107")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(classes, rotation=45, ha="right")
            axes[0].set_title("Verification Results by Class")
            axes[0].legend()
            axes[0].set_ylabel("Count")

            # Confusion matrix (mislabel heatmap)
            all_classes = sorted(set(
                [r.class_name for r in self.results] +
                [r.qwen_class for r in self.results if r.qwen_class and r.is_correct is False]
            ))
            n = len(all_classes)
            matrix = np.zeros((n, n), dtype=int)
            cls_to_idx = {c: i for i, c in enumerate(all_classes)}

            for r in self.results:
                if r.qwen_class and r.qwen_class in cls_to_idx:
                    labeled_idx = cls_to_idx.get(r.class_name, -1)
                    qwen_idx = cls_to_idx.get(r.qwen_class, -1)
                    if labeled_idx >= 0 and qwen_idx >= 0:
                        matrix[labeled_idx][qwen_idx] += 1

            im = axes[1].imshow(matrix, cmap="YlOrRd", aspect="auto")
            axes[1].set_xticks(range(n))
            axes[1].set_yticks(range(n))
            axes[1].set_xticklabels(all_classes, rotation=45, ha="right", fontsize=8)
            axes[1].set_yticklabels(all_classes, fontsize=8)
            axes[1].set_xlabel("Qwen Predicted")
            axes[1].set_ylabel("Dataset Label")
            axes[1].set_title("Label vs Qwen Prediction")
            for i in range(n):
                for j in range(n):
                    if matrix[i][j] > 0:
                        axes[1].text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=axes[1])

            plt.tight_layout()
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img

        except ImportError:
            return None

    def _get_mismatch_gallery(self) -> list:
        """Get gallery of mismatch crops for visualization."""
        mismatches = [r for r in self.results if r.is_correct is False and r.crop_path]
        gallery = []
        for r in mismatches[:50]:
            caption = f"Label: {r.class_name} → Qwen: {r.qwen_class} ({r.qwen_confidence})"
            gallery.append((r.crop_path, caption))
        return gallery if gallery else []

    def get_results_table(self) -> list:
        """Get results as table data for Gradio Dataframe."""
        rows = []
        for i, r in enumerate(self.results):
            status = "✅" if r.is_correct is True else ("❌" if r.is_correct is False else "❓")
            rows.append([
                i,
                status,
                os.path.basename(r.image_path),
                r.line_idx,
                r.class_name,
                r.qwen_class or "N/A",
                r.qwen_confidence or "N/A",
                r.qwen_reasoning or "",
            ])
        return rows

    def get_mismatch_table(self) -> list:
        """Get only mismatches for fixing."""
        rows = []
        for i, r in enumerate(self.results):
            if r.is_correct is False:
                rows.append([
                    i,
                    os.path.basename(r.image_path),
                    r.line_idx,
                    r.class_name,
                    r.qwen_class or "N/A",
                    r.qwen_confidence or "N/A",
                    r.qwen_reasoning or "",
                ])
        return rows

    def preview_entry(self, result_idx: int) -> tuple:
        """Preview original image with bbox and crop for a specific result."""
        if result_idx < 0 or result_idx >= len(self.results):
            return None, None, "Invalid index"

        r = self.results[result_idx]
        # Draw bbox on original
        img = cv2.imread(r.image_path)
        if img is None:
            return None, None, "Cannot read image"

        h, w = img.shape[:2]
        x1 = int((r.cx - r.w / 2) * w)
        y1 = int((r.cy - r.h / 2) * h)
        x2 = int((r.cx + r.w / 2) * w)
        y2 = int((r.cy + r.h / 2) * h)

        color = (0, 0, 255) if r.is_correct is False else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label_text = f"Label: {r.class_name}"
        if r.qwen_class:
            label_text += f" | Qwen: {r.qwen_class}"
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop
        crop = self.crop_object(r)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop is not None else None

        info = (
            f"Image: {os.path.basename(r.image_path)}\n"
            f"Label line: {r.line_idx}\n"
            f"Dataset label: {r.class_name} (id={r.class_id})\n"
            f"Qwen prediction: {r.qwen_class}\n"
            f"Confidence: {r.qwen_confidence}\n"
            f"Reason: {r.qwen_reasoning}"
        )
        return img_rgb, crop_rgb, info

    def fix_label(self, result_idx: int, new_class: str, action: str) -> str:
        """Fix a label: change class or delete the annotation line."""
        if result_idx < 0 or result_idx >= len(self.results):
            return "Invalid index"

        r = self.results[result_idx]
        lbl_path = r.label_path

        # Read current labels
        with open(lbl_path, "r") as f:
            lines = f.readlines()

        if action == "delete":
            # Remove this annotation line
            if r.line_idx < len(lines):
                backup_path = lbl_path + ".bak"
                shutil.copy2(lbl_path, backup_path)
                del lines[r.line_idx]
                with open(lbl_path, "w") as f:
                    f.writelines(lines)
                return f"✅ Deleted line {r.line_idx} from {os.path.basename(lbl_path)} (backup saved)"
            return "❌ Line index out of range"

        elif action == "relabel":
            # Change class
            if new_class not in self.dataset.class_names:
                return f"❌ Unknown class: {new_class}. Valid: {self.dataset.class_names}"
            new_class_id = self.dataset.class_names.index(new_class)

            if r.line_idx < len(lines):
                backup_path = lbl_path + ".bak"
                shutil.copy2(lbl_path, backup_path)
                parts = lines[r.line_idx].strip().split()
                parts[0] = str(new_class_id)
                lines[r.line_idx] = " ".join(parts) + "\n"
                with open(lbl_path, "w") as f:
                    f.writelines(lines)
                # Update in-memory
                r.class_id = new_class_id
                r.class_name = new_class
                return f"✅ Relabeled line {r.line_idx} to '{new_class}' in {os.path.basename(lbl_path)} (backup saved)"
            return "❌ Line index out of range"

        return "❌ Unknown action"

    def batch_fix(self, apply_qwen_suggestions: bool = False, min_confidence: str = "HIGH") -> str:
        """Batch fix: apply all Qwen suggestions or delete low-confidence entries."""
        if not self.results:
            return "No results to fix."

        mismatches = [r for r in self.results if r.is_correct is False]
        if not mismatches:
            return "No mismatches found. Dataset looks clean!"

        conf_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        min_conf_val = conf_order.get(min_confidence, 0)

        fixed = 0
        skipped = 0
        errors = []

        for r in mismatches:
            r_conf = conf_order.get(r.qwen_confidence, 0)
            if r_conf < min_conf_val:
                skipped += 1
                continue

            if apply_qwen_suggestions and r.qwen_class in self.dataset.class_names:
                new_id = self.dataset.class_names.index(r.qwen_class)
                try:
                    with open(r.label_path, "r") as f:
                        lines = f.readlines()
                    if r.line_idx < len(lines):
                        backup_path = r.label_path + ".bak"
                        if not os.path.exists(backup_path):
                            shutil.copy2(r.label_path, backup_path)
                        parts = lines[r.line_idx].strip().split()
                        parts[0] = str(new_id)
                        lines[r.line_idx] = " ".join(parts) + "\n"
                        with open(r.label_path, "w") as f:
                            f.writelines(lines)
                        r.class_id = new_id
                        r.class_name = r.qwen_class
                        r.is_correct = True
                        fixed += 1
                except Exception as e:
                    errors.append(f"{os.path.basename(r.label_path)}: {e}")

        return (
            f"Batch fix complete:\n"
            f"  Fixed: {fixed}\n"
            f"  Skipped (low confidence): {skipped}\n"
            f"  Errors: {len(errors)}\n"
            + ("\n".join(errors[:10]) if errors else "")
        )

    def export_results(self, output_path: str) -> str:
        """Export verification results to JSON."""
        if not self.results:
            return "No results to export."

        data = []
        for r in self.results:
            data.append({
                "image": r.image_path,
                "label_file": r.label_path,
                "line_idx": r.line_idx,
                "class_id": r.class_id,
                "class_name": r.class_name,
                "bbox": [r.cx, r.cy, r.w, r.h],
                "qwen_class": r.qwen_class,
                "qwen_confidence": r.qwen_confidence,
                "qwen_reasoning": r.qwen_reasoning,
                "is_correct": r.is_correct,
            })

        output_path = output_path.strip() or os.path.join(self.dataset.base_dir, "verification_results.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return f"✅ Exported {len(data)} results to {output_path}"


# ──────────────────────────────────────────────
# Gradio App
# ──────────────────────────────────────────────
def build_app():
    checker = DatasetChecker()

    with gr.Blocks(title="YOLO Dataset Checker (Qwen VL)", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔍 YOLO Dataset Checker with Qwen Vision Model")
        gr.Markdown(
            "Verify your YOLO object detection labels using Qwen VL model. "
            "Crops each annotated object and asks Qwen to verify the class."
        )

        # ──── Tab 1: Setup ────
        with gr.Tab("1. Setup"):
            gr.Markdown("### API Configuration")
            gr.Markdown(
                "Connect to Qwen VL via OpenAI-compatible API. Supports:\n"
                "- **Dashscope**: `https://dashscope.aliyuncs.com/compatible-mode/v1` (model: `qwen-vl-plus` or `qwen-vl-max`)\n"
                "- **Ollama**: `http://localhost:11434/v1` (model: `qwen2.5-vl` etc.)\n"
                "- **vLLM**: `http://localhost:8000/v1`\n"
                "- **OpenRouter**: `https://openrouter.ai/api/v1`"
            )
            with gr.Row():
                api_key = gr.Textbox(label="API Key", type="password", placeholder="sk-...")
                base_url = gr.Textbox(
                    label="Base URL",
                    value="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    placeholder="https://..."
                )
                model_name = gr.Textbox(label="Model Name", value="qwen-vl-plus")
            connect_btn = gr.Button("Connect", variant="primary")
            connect_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### Dataset Configuration")
            with gr.Row():
                yaml_input = gr.Textbox(
                    label="data.yaml Path",
                    value="/home/sonnn/Videos/tmp_dataset/pvn-dataset/pvn-vehicle/251222/data.yaml",
                    placeholder="/path/to/data.yaml",
                )
                split_select = gr.Dropdown(
                    label="Split", choices=["train", "val", "test"], value="val"
                )
                max_images = gr.Number(label="Max Images (0=all)", value=50, precision=0)
            load_btn = gr.Button("Load Dataset", variant="primary")
            load_status = gr.Textbox(label="Dataset Info", interactive=False, lines=10)

            connect_btn.click(
                checker.connect_api,
                inputs=[api_key, base_url, model_name],
                outputs=connect_status,
            )
            load_btn.click(
                checker.load_dataset,
                inputs=[yaml_input, split_select, max_images],
                outputs=load_status,
            )

        # ──── Tab 2: Verify ────
        with gr.Tab("2. Verify"):
            gr.Markdown("### Run Verification")
            with gr.Row():
                max_objects = gr.Number(label="Max Objects to Check (0=all)", value=100, precision=0)
                classes_filter = gr.Textbox(
                    label="Classes to Check (comma-separated, empty=all)",
                    placeholder="car, bus, truck",
                )
            verify_btn = gr.Button("🚀 Start Verification", variant="primary")
            report_text = gr.Textbox(label="Verification Report", interactive=False, lines=25)

            with gr.Row():
                summary_chart = gr.Image(label="Summary Chart", type="numpy")
            mismatch_gallery = gr.Gallery(
                label="❌ Mismatched Labels", columns=5, height=400
            )

            def run_verify(max_obj, classes_str, progress=gr.Progress()):
                classes = [c.strip() for c in classes_str.split(",") if c.strip()] if classes_str.strip() else None
                report, chart, gallery = checker.run_verification(
                    max_objects=int(max_obj),
                    classes_to_check=classes,
                    progress=progress,
                )
                return report, chart, gallery

            verify_btn.click(
                run_verify,
                inputs=[max_objects, classes_filter],
                outputs=[report_text, summary_chart, mismatch_gallery],
            )

        # ──── Tab 3: Analyze ────
        with gr.Tab("3. Analyze & Browse"):
            gr.Markdown("### Browse Results")
            with gr.Row():
                result_filter = gr.Radio(
                    label="Filter",
                    choices=["All", "Correct", "Incorrect", "Unknown"],
                    value="All",
                )
                refresh_btn = gr.Button("Refresh Table")

            results_table = gr.Dataframe(
                headers=["Idx", "Status", "Image", "Line", "Label", "Qwen Says", "Confidence", "Reason"],
                interactive=False,
                wrap=True,
            )

            def get_filtered_table(filter_val):
                rows = checker.get_results_table()
                if filter_val == "Correct":
                    rows = [r for r in rows if r[1] == "✅"]
                elif filter_val == "Incorrect":
                    rows = [r for r in rows if r[1] == "❌"]
                elif filter_val == "Unknown":
                    rows = [r for r in rows if r[1] == "❓"]
                return rows

            refresh_btn.click(get_filtered_table, inputs=[result_filter], outputs=[results_table])

            gr.Markdown("---")
            gr.Markdown("### Preview Entry")
            with gr.Row():
                preview_idx = gr.Number(label="Result Index", value=0, precision=0)
                preview_btn = gr.Button("Preview")
            with gr.Row():
                preview_original = gr.Image(label="Original Image (with bbox)", type="numpy")
                preview_crop = gr.Image(label="Cropped Object", type="numpy")
            preview_info = gr.Textbox(label="Details", interactive=False, lines=6)

            preview_btn.click(
                lambda idx: checker.preview_entry(int(idx)),
                inputs=[preview_idx],
                outputs=[preview_original, preview_crop, preview_info],
            )

        # ──── Tab 4: Fix ────
        with gr.Tab("4. Fix Labels"):
            gr.Markdown("### Fix Individual Label")
            with gr.Row():
                fix_idx = gr.Number(label="Result Index", value=0, precision=0)
                fix_action = gr.Radio(
                    label="Action", choices=["relabel", "delete"], value="relabel"
                )
                fix_new_class = gr.Textbox(label="New Class (for relabel)", placeholder="car")
            fix_btn = gr.Button("Apply Fix", variant="primary")
            fix_status = gr.Textbox(label="Fix Status", interactive=False)

            fix_btn.click(
                lambda idx, cls, act: checker.fix_label(int(idx), cls, act),
                inputs=[fix_idx, fix_new_class, fix_action],
                outputs=[fix_status],
            )

            gr.Markdown("---")
            gr.Markdown("### Batch Fix (Apply Qwen Suggestions)")
            with gr.Row():
                batch_apply = gr.Radio(
                    label="Apply Qwen class suggestions",
                    choices=["Yes", "No"], value="Yes",
                )
                batch_min_conf = gr.Dropdown(
                    label="Min Confidence", choices=["HIGH", "MEDIUM", "LOW"], value="HIGH"
                )
            batch_btn = gr.Button("🔧 Batch Fix", variant="stop")
            batch_status = gr.Textbox(label="Batch Fix Status", interactive=False, lines=5)

            batch_btn.click(
                lambda apply_s, min_c: checker.batch_fix(apply_s == "Yes", min_c),
                inputs=[batch_apply, batch_min_conf],
                outputs=[batch_status],
            )

        # ──── Tab 5: Export ────
        with gr.Tab("5. Export"):
            gr.Markdown("### Export Results")
            export_path = gr.Textbox(
                label="Output Path (empty=auto)",
                placeholder="/path/to/verification_results.json",
            )
            export_btn = gr.Button("Export to JSON", variant="primary")
            export_status = gr.Textbox(label="Export Status", interactive=False)

            export_btn.click(
                checker.export_results,
                inputs=[export_path],
                outputs=[export_status],
            )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7865, share=False)
