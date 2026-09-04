from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str):
    checks.append((name, condition, detail))


check("Python", sys.version_info >= (3, 10), f"{platform.python_version()} ({platform.system()} {platform.machine()})")
for module in ("numpy", "torch", "PIL"):
    try:
        loaded = importlib.import_module(module)
        check(module, True, getattr(loaded, "__version__", "available"))
    except ImportError as error:
        check(module, False, str(error))

model_path = ROOT / "runs/shrec17_benchmark/stgcn_full/best.pt"
check("14-class checkpoint", model_path.exists(), str(model_path.relative_to(ROOT)))
check("MediaPipe browser model", (ROOT / "assets/models/hand_landmarker.task").exists(), "assets/models/hand_landmarker.task")
check("MediaPipe JS bundle", (ROOT / "node_modules/@mediapipe/tasks-vision/vision_bundle.mjs").exists(), "run npm install if missing")
references = list((ROOT / "assets/references").glob("*.gif"))
check("Reference animations", len(references) == 14, f"{len(references)} / 14")

if model_path.exists():
    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        labels = checkpoint.get("labels", [])
        check("Checkpoint labels", len(labels) == 14, f"{len(labels)} labels")
    except Exception as error:  # noqa: BLE001 - diagnostic tool should report, not crash.
        check("Checkpoint load", False, str(error))

print("GestureGraph Lab - environment check")
print("=" * 42)
for name, passed, detail in checks:
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
failed = [name for name, passed, _ in checks if not passed]
print("=" * 42)
print(json.dumps({"passed": len(checks) - len(failed), "failed": failed}, ensure_ascii=False))
raise SystemExit(1 if failed else 0)
