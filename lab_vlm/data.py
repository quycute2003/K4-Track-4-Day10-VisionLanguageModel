"""Labels, prompts, deterministic batches and reward (no GPU required)."""
import hashlib
import json
import math
import random
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import iou_xyxy, parse_output


def make_prompt(coord="norm1000", width=640, height=360):
    if coord not in ("norm1000", "pixel"):
        raise ValueError(coord)
    scale = ("integer coordinates normalized to 0–1000 relative to image width and height"
             if coord == "norm1000" else
             f"integer pixel coordinates in the original {width}×{height} image")
    return (
        "Locate every visible road defect in this dashcam image. "
        "Use only these labels: pothole, crack, manhole. "
        f"Return bounding boxes using {scale}. "
        'Return ONLY a JSON list with objects of the form '
        '{"label":"pothole","bbox_2d":[x1,y1,x2,y2]}. '
        "Use the top-left and bottom-right corners. "
        "Sort objects from top to bottom, then left to right. "
        "Return [] when no defect is visible. Do not add Markdown or explanations."
    )


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    if label not in CLASSES or len(yolo) != 4:
        raise ValueError("Unknown class or invalid YOLO box")
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in yolo):
        raise ValueError("YOLO coordinates must be finite and in [0,1]")
    if coord not in ("norm1000", "pixel"):
        raise ValueError(coord)
    xc, yc, w, h = yolo
    if w <= 0 or h <= 0:
        raise ValueError("Zero-area annotation cannot form a valid bounding-box target")
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    limits = (sx, sy, sx, sy)
    corners = ((xc-w/2)*sx, (yc-h/2)*sy, (xc+w/2)*sx, (yc+h/2)*sy)
    bb = [min(limit, max(0, round(v))) for v, limit in zip(corners, limits)]
    # Keep ordinary rounding unchanged. For a positive but sub-unit extent, use
    # outward rounding only on the collapsed axis; retain bounds and positive area.
    for lower, upper, limit in ((0, 2, sx), (1, 3, sy)):
        if bb[upper] <= bb[lower]:
            bb[lower] = min(limit-1, max(0, math.floor(corners[lower])))
            bb[upper] = min(limit, max(bb[lower]+1, math.ceil(corners[upper])))
    return {"label": label, "bbox_2d": bb}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    boxes = []
    for obj in objects:
        yolo = obj["yolo"]
        # The source dataset contains one crack annotation with h=0. Fabricating
        # an extent would change that annotation; omit it from SFT targets instead.
        if len(yolo) == 4 and (yolo[2] == 0 or yolo[3] == 0):
            continue
        boxes.append(yolo_to_box(obj["label"], yolo, coord, width, height))
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0], b["label"], b["bbox_2d"]))
    return json.dumps(boxes, separators=(",", ":"), ensure_ascii=False)


def target_quality_report(entries, coord="norm1000"):
    """Validate every target before allocating a model; record invalid source boxes."""
    skipped, repairs, total = [], 0, 0
    for entry in entries:
        objects_to_answer(entry["objects"], coord, entry["width"], entry["height"])
        for obj in entry["objects"]:
            total += 1
            xc, yc, w, h = obj["yolo"]
            if w == 0 or h == 0:
                skipped.append({"image_id": entry["image_id"], "label": obj["label"], "yolo": obj["yolo"]})
                continue
            sx, sy = (1000, 1000) if coord == "norm1000" else (entry["width"], entry["height"])
            raw = ((xc-w/2)*sx, (yc-h/2)*sy, (xc+w/2)*sx, (yc+h/2)*sy)
            bb = [min(limit, max(0, round(v))) for v, limit in zip(raw, (sx, sy, sx, sy))]
            repairs += int(bb[2] <= bb[0] or bb[3] <= bb[1])
    return {"source_objects": total, "target_objects": total-len(skipped),
            "zero_area_annotations_skipped": len(skipped), "zero_area_annotations": skipped,
            "positive_boxes_repaired_after_rounding": repairs}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def verify_splits(data_dir="data"):
    splits = {s: read_jsonl(Path(data_dir) / "splits" / f"{s}.jsonl")
              for s in ("train", "val", "test")}
    sessions = {s: {r["session"] for r in rows} for s, rows in splits.items()}
    ids = {s: {r["image_id"] for r in rows} for s, rows in splits.items()}
    for s, rows in splits.items():
        if not rows or len(ids[s]) != len(rows):
            raise ValueError(f"Empty split or duplicate image IDs: {s}")
        for other in splits:
            if s != other and (sessions[s] & sessions[other] or ids[s] & ids[other]):
                raise ValueError(f"Data leakage: {s}/{other}")
    meta = json.loads((Path(data_dir) / "splits" / "meta.json").read_text(encoding="utf-8"))
    for s, rows in splits.items():
        counts = {c: sum(o["label"] == c for r in rows for o in r["objects"]) for c in CLASSES}
        actual = {"images": len(rows), "boxes": counts}
        if actual != meta["counts"][s]:
            raise ValueError(f"Split metadata mismatch: {s}")
        print(s, actual)
    return splits, meta


def select_train(rows, fraction=1.0, seed=42):
    if not 0 < fraction <= 1:
        raise ValueError("train_fraction must be in (0,1]")
    # Nested subsets; a fixed shuffle prevents 10% from selecting one contiguous session.
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    return ordered[:max(1, int(len(ordered) * fraction))]


def batch_indices(n, batch_size, micro_step, seed=42):
    """Deterministic epoch permutations; resume needs only the micro-step number."""
    if n <= 0 or batch_size <= 0 or micro_step < 0:
        raise ValueError("Invalid batch arguments")
    batches = math.ceil(n / batch_size)
    epoch, index = divmod(micro_step, batches)
    order = list(range(n))
    random.Random(seed + epoch).shuffle(order)
    return order[index * batch_size:(index + 1) * batch_size]


def reward_iou(completion, gt_boxes):
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    gt = [{"label": b["label"], "bbox": b.get("bbox", b.get("bbox_2d"))} for b in gt_boxes]
    pairs = sorted((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                   for i, g in enumerate(gt) for j, p in enumerate(pred) if g["label"] == p["label"])
    used_g, used_p, total = set(), set(), 0.0
    for value, i, j in reversed(pairs):
        if value >= 0.5 and i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            total += value
    return 0.2 + 0.8 * total / max(len(gt), len(pred), 1)
