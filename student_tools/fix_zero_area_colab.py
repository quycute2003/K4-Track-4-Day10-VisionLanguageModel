"""Hotfix older Colab packages, preserving source annotations and existing checkpoints.

Execute in the notebook before rerunning cell 6:
    %run student_tools/fix_zero_area_colab.py
The updated package already has the fix; this tool is idempotent.
"""
import gc
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
path = ROOT/"lab_vlm/data.py"
text = path.read_text(encoding="utf-8")
old = '    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]'
new = '''    boxes = [
        yolo_to_box(o["label"], o["yolo"], coord, width, height)
        for o in objects
        if o["yolo"][2] > 0 and o["yolo"][3] > 0
    ]'''
if old in text:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
elif new not in text and 'Zero-area annotation' not in text:
    raise RuntimeError("Source code differs; inspect objects_to_answer before patching")

from lab_vlm import data, encoding
importlib.reload(data)
importlib.reload(encoding)
bad = []
for split in ("train", "val", "test"):
    split_file = ROOT/"data/splits"/f"{split}.jsonl"
    if not split_file.exists():
        continue
    for entry in data.read_jsonl(split_file):
        data.objects_to_answer(entry["objects"], width=entry["width"], height=entry["height"])
        bad.extend({"split": split, "image_id": entry["image_id"], **obj} for obj in entry["objects"]
                   if obj["yolo"][2] == 0 or obj["yolo"][3] == 0)
data.write_json(ROOT/"report/diagnostics/zero_area_labels.json", {"omitted_from_targets": bad})
print(f"Targets validated. Zero-area annotations omitted: {len(bad)}", flush=True)

# IPython may hold the model alive through the previous exception's traceback.
for name in ("last_value", "last_exc"):
    error = getattr(sys, name, None)
    if isinstance(error, BaseException):
        error.__traceback__ = None
sys.last_traceback = None
gc.collect()
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Rerun cell 6; source labels and previously saved checkpoints are preserved.", flush=True)
