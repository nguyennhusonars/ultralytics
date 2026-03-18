"""
Script to calculate model info (layers, params, gradients, GFLOPs) for all YAML model configs
under ultralytics/cfg/models/ and save results to a CSV file.

Uses a **persistent worker subprocess** that stays alive across many models (fast — no
re-import overhead). If the worker crashes (segfault), it is automatically restarted.
Results are written to the CSV incrementally so nothing is lost.

Usage:
    python calc_model_info.py [--imgsz 640] [--output model_info_results.csv]
    python calc_model_info.py --models-dir ultralytics/cfg/models/11
    python calc_model_info.py --resume          # skip model variants already in the CSV
    python calc_model_info.py --timeout 120     # per-model timeout in seconds
"""

import argparse
import csv
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
FIELDNAMES = [
    "yaml_file", "model_name", "scale", "task", "nc",
    "layers", "parameters", "gradients", "gflops", "error",
]
_SENTINEL = "__DONE__"


# ── helpers (parent process) ─────────────────────────────────────────────

def find_all_yaml_configs(models_dir: Path) -> list:
    return sorted(models_dir.rglob("*.yaml"))


def get_task_from_yaml(cfg: dict, yaml_path: Path) -> str:
    try:
        m = cfg["head"][-1][-2].lower()
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if "pose" in m:
            return "pose"
        if "obb" in m:
            return "obb"
    except (KeyError, IndexError, TypeError):
        pass
    stem = yaml_path.stem.lower()
    parts = [p.lower() for p in yaml_path.parts]
    if "-seg" in stem or "segment" in parts:
        return "segment"
    if "-cls" in stem or "classify" in parts:
        return "classify"
    if "-pose" in stem or "pose" in parts:
        return "pose"
    if "-obb" in stem or "obb" in parts:
        return "obb"
    return "detect"


def yaml_jobs(yaml_path: Path, models_dir: Path) -> list:
    """Return a list of (yaml_path_str, rel_path, task, nc, scale_name) jobs."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if not cfg or "backbone" not in cfg or "head" not in cfg:
        return []
    task = get_task_from_yaml(cfg, yaml_path)
    nc = cfg.get("nc", 80)
    rel_path = str(yaml_path.relative_to(models_dir))
    scales = cfg.get("scales")
    if scales:
        return [(str(yaml_path), rel_path, task, nc, s) for s in scales]
    return [(str(yaml_path), rel_path, task, nc, None)]


# ── persistent worker (child process) ────────────────────────────────────

def _worker_loop(job_queue, result_queue):
    """Long-running worker: read jobs from job_queue, send results to result_queue."""
    import logging
    import warnings
    from collections import OrderedDict

    # Suppress all noisy output
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    # Redirect stderr to devnull to silence C-level warnings
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)

    import torch
    import torch.nn as nn

    # Patch missing module references
    _Stub = type("_Stub", (nn.Module,), {})
    import ultralytics.nn.tasks as _tasks
    for _name in ["XSSBlock"]:
        if not hasattr(_tasks, _name):
            _tasks.__dict__[_name] = _Stub

    from ultralytics.nn.tasks import (
        ClassificationModel, DetectionModel, OBBModel,
        PoseModel, SegmentationModel, yaml_model_load,
    )
    from ultralytics.utils.torch_utils import get_num_gradients, get_num_params

    def get_flops_safe(model, imgsz=640):
        """Calculate GFLOPs using model.yaml['channels'] instead of first param shape."""
        try:
            import thop
        except ImportError:
            return 0.0
        try:
            from ultralytics.utils.torch_utils import unwrap_model
            from copy import deepcopy
            model = unwrap_model(model)
            ch = model.yaml.get("channels", 3) if hasattr(model, "yaml") else 3
            if not isinstance(imgsz, list):
                imgsz = [imgsz, imgsz]
            try:
                stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32
                im = torch.empty((1, ch, stride, stride), device=next(model.parameters()).device)
                flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2
                return flops * imgsz[0] / stride * imgsz[1] / stride
            except Exception:
                im = torch.empty((1, ch, *imgsz), device=next(model.parameters()).device)
                return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2
        except Exception:
            return 0.0

    task_map = {
        "detect": DetectionModel, "segment": SegmentationModel,
        "classify": ClassificationModel, "pose": PoseModel, "obb": OBBModel,
    }

    # Signal parent that we're ready
    result_queue.put(_SENTINEL)

    while True:
        job = job_queue.get()
        if job is None:  # poison pill
            break

        yaml_path_str, task, nc, scale_name, imgsz = job
        yaml_path = Path(yaml_path_str)
        ModelClass = task_map.get(task, DetectionModel)
        model_name = f"{yaml_path.stem}-{scale_name}" if scale_name else yaml_path.stem

        try:
            if scale_name:
                d = yaml_model_load(yaml_path)
                d["scale"] = scale_name
                if d.get("scales"):
                    for k, v in d["scales"].items():
                        if len(v) == 3:
                            d["scales"][k] = list(v) + [0]
                cfg_arg = d
            else:
                cfg_arg = yaml_model_load(yaml_path)

            # Inject default kpt_shape for pose models if missing
            if task == "pose" and isinstance(cfg_arg, dict):
                cfg_arg.setdefault("kpt_shape", [17, 3])

            with torch.no_grad():
                m = ModelClass(cfg=cfg_arg, nc=nc, verbose=False)

            layers = OrderedDict((n, mod) for n, mod in m.named_modules() if len(mod._modules) == 0)
            n_l = len(layers)
            n_p = get_num_params(m)
            n_g = get_num_gradients(m)
            flops = get_flops_safe(m, imgsz)
            del m

            result_queue.put({
                "model_name": model_name,
                "scale": scale_name or "-",
                "layers": n_l,
                "parameters": n_p,
                "gradients": n_g,
                "gflops": round(flops, 1) if flops else 0,
            })
        except Exception as e:
            result_queue.put({
                "model_name": model_name,
                "scale": scale_name or "-",
                "layers": "ERR", "parameters": "ERR",
                "gradients": "ERR", "gflops": "ERR",
                "error": str(e) or repr(e),
            })


class WorkerManager:
    """Manages a persistent worker subprocess, restarting on crash."""

    def __init__(self, timeout: int):
        self.timeout = timeout
        self.ctx = mp.get_context("spawn")
        self._start_worker()

    def _start_worker(self):
        self.job_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.proc = self.ctx.Process(target=_worker_loop, args=(self.job_q, self.result_q))
        self.proc.start()
        # Wait for worker to signal ready
        try:
            msg = self.result_q.get(timeout=120)
            assert msg == _SENTINEL
        except Exception:
            raise RuntimeError("Worker failed to start")

    def run(self, yaml_path_str, rel_path, task, nc, scale_name, imgsz) -> dict:
        model_name = f"{Path(yaml_path_str).stem}-{scale_name}" if scale_name else Path(yaml_path_str).stem

        self.job_q.put((yaml_path_str, task, nc, scale_name, imgsz))

        # Wait for result with timeout
        start = time.monotonic()
        while time.monotonic() - start < self.timeout:
            if not self.proc.is_alive():
                # Worker crashed (segfault etc.)
                self.proc.join()
                exitcode = self.proc.exitcode
                self._start_worker()
                return self._error_result(
                    rel_path, model_name, scale_name, task, nc,
                    f"Worker crashed (exit code {exitcode})"
                )
            try:
                result = self.result_q.get(timeout=1)
                result["yaml_file"] = rel_path
                result["task"] = task
                result["nc"] = nc
                return result
            except Exception:
                continue

        # Timeout — kill and restart the worker
        self.proc.kill()
        self.proc.join()
        self._start_worker()
        return self._error_result(
            rel_path, model_name, scale_name, task, nc,
            f"Timeout after {self.timeout}s"
        )

    def shutdown(self):
        if self.proc.is_alive():
            self.job_q.put(None)  # poison pill
            self.proc.join(timeout=5)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join()

    @staticmethod
    def _error_result(rel_path, model_name, scale_name, task, nc, error):
        return {
            "yaml_file": rel_path, "model_name": model_name,
            "scale": scale_name or "-", "task": task, "nc": nc,
            "layers": "ERR", "parameters": "ERR",
            "gradients": "ERR", "gflops": "ERR",
            "error": error,
        }


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Calculate model info for all YAML configs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for GFLOPs calc")
    parser.add_argument("--output", type=str, default="model_info_results.csv", help="Output CSV")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to models config dir")
    parser.add_argument("--timeout", type=int, default=120, help="Per-model timeout in seconds")
    parser.add_argument("--resume", action="store_true", help="Skip model variants already in CSV")
    args = parser.parse_args()

    models_dir = Path(args.models_dir) if args.models_dir else ROOT / "ultralytics" / "cfg" / "models"
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        sys.exit(1)

    output_path = Path(args.output)

    # Collect already-done model variants when resuming
    done_keys = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_keys.add((row["yaml_file"], row.get("scale", "-")))
        print(f"Resuming: {len(done_keys)} model variants already in {output_path}")

    # Build job list
    yaml_files = find_all_yaml_configs(models_dir)
    print(f"Found {len(yaml_files)} YAML config files under {models_dir}")

    all_jobs = []
    for yp in yaml_files:
        try:
            all_jobs.extend(yaml_jobs(yp, models_dir))
        except Exception:
            rel = str(yp.relative_to(models_dir))
            all_jobs.append((str(yp), rel, "unknown", 80, None))

    # Filter out already-done jobs
    if done_keys:
        all_jobs = [j for j in all_jobs if (j[1], j[4] or "-") not in done_keys]

    print(f"Total model variants: {len(all_jobs)}\n")

    if not all_jobs:
        print("Nothing to do.")
        return

    # Open CSV
    if args.resume and output_path.exists():
        csv_file = open(output_path, "a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES, extrasaction="ignore")
    else:
        csv_file = open(output_path, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        csv_file.flush()

    # Start persistent worker
    print("Starting worker...", flush=True)
    manager = WorkerManager(timeout=args.timeout)
    print("Worker ready.\n", flush=True)

    success_count = 0
    error_count = 0

    try:
        for i, (yp_str, rel_path, task, nc, scale_name) in enumerate(all_jobs, 1):
            label = f"{Path(yp_str).stem}-{scale_name}" if scale_name else Path(yp_str).stem
            print(f"[{i}/{len(all_jobs)}] {label}", end=" ... ", flush=True)

            result = manager.run(yp_str, rel_path, task, nc, scale_name, args.imgsz)

            if result.get("error") or result.get("layers") == "ERR":
                error_count += 1
                print(f"FAIL: {result.get('error', 'unknown error')}")
            else:
                success_count += 1
                p = result['parameters']
                p_str = f"{p:,}" if isinstance(p, (int, float)) else str(p)
                print(f"OK  {result['layers']} layers, {p_str} params, {result['gflops']} GFLOPs")

            writer.writerow(result)
            csv_file.flush()
    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Partial results saved.")
    finally:
        manager.shutdown()
        csv_file.close()

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path.resolve()}")
    print(f"Total: {success_count + error_count} | Success: {success_count} | Errors: {error_count}")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
