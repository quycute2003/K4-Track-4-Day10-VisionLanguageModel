"""Generate measured tables and annotated images without inventing report answers."""
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from scripts.labcommon import read_jsonl
from scripts.score import evaluate, parse_output
from .data import objects_to_answer


def evidence_table(evidence_dir="report/evidence"):
    rows = []
    for path in sorted(Path(evidence_dir).glob("e*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("split") != "test":
            continue
        m = d.get("meta", {})
        ci = d["f1_ci95"]
        setup = m.get("setup", d["name"])
        setup = {"E1": "LoRA LLM", "E2": "+ projector", "E3": "+ LoRA ViT"}.get(setup, setup)
        resource = lambda key: f"{m[key]:.2f}" if isinstance(m.get(key), (int, float)) else "—"
        rows.append(f"| {d['name']} | {setup} | {m.get('trainable_params', 0)} | "
                    f"{resource('peak_vram_gb')} | {resource('train_minutes')} | "
                    f"{d['micro']['f1']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {d['fp_per_image']:.4f} | "
                    + " | ".join(f"{d['per_class'][c]['f1']:.4f}" for c in ("pothole", "crack", "manhole")) + " |")
    return ("| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))


def make_summary(evidence_dir="report/evidence", out="report/SUMMARY.md", report_path=None):
    table = evidence_table(evidence_dir)
    if table.count("\n") < 2:
        raise ValueError("No test evidence available")
    has_e4 = any(Path(evidence_dir).glob("e4*.json"))
    experiment_note = ("E4 chọn setup bằng F1 trên validation. " if has_e4 else
                       "Chưa có evidence E4; bảng chỉ liệt kê các thí nghiệm đã được chấm. ")
    text = ("# Kết quả đo từ các lần chạy thật\n\n" + table +
            "\n\nCI95 được bootstrap theo ảnh bởi công cụ chấm của đề. "
            "CI của từng F1 không phải CI của chênh lệch giữa hai mô hình; "
            "không kết luận ý nghĩa thống kê chỉ từ việc hai khoảng chồng lấp.\n\n"
            + experiment_note + "Xem nhật ký checkpoint để lấy "
            "số module, số token ảnh, tham số từng khối và loss. "
            "Xem REPORT.md để đọc phần phân tích và 10 câu trả lời.\n")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    if report_path:
        path = Path(report_path)
        current = path.read_text(encoding="utf-8")
        updated, count = re.subn(r"(?m)^\| Mã \|[^\n]*\n\|[-| ]+\|\n(?:\|[^\n]*\n?)+", table+"\n", current, count=1)
        if count != 1:
            raise ValueError("Could not find the summary table in REPORT.md")
        versions = {json.loads(p.read_text(encoding="utf-8"))["dataset_version"]
                    for p in Path(evidence_dir).glob("e*.json")}
        if len(versions) != 1:
            raise ValueError("Mixed dataset versions")
        updated = re.sub(r"(?m)^`dataset_version`:.*$", f"`dataset_version`: {next(iter(versions))}", updated)
        path.write_text(updated, encoding="utf-8")
    print(text)
    return Path(out)


def _font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_boxes(image, boxes, color, coord, width, height):
    draw = ImageDraw.Draw(image)
    sx, sy = (width/1000, height/1000) if coord == "norm1000" else (1, 1)
    for b in boxes:
        bb = b.get("bbox", b.get("bbox_2d"))
        corners = [bb[0]*sx, bb[1]*sy, bb[2]*sx, bb[3]*sy]
        draw.rectangle(corners, outline=color, width=2)
        draw.text((max(0, corners[0]), max(0, corners[1]-15)), b["label"], fill=color, font=_font(12))


def ground_truth_preview(entries, data_dir="data", out_dir="report/figures/labels", count=6):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, entry in enumerate(entries[:count]):
        with Image.open(Path(data_dir)/entry["image_path"]) as source:
            image = source.convert("RGB")
        boxes = json.loads(objects_to_answer(entry["objects"]))
        _draw_boxes(image, boxes, "lime", "norm1000", entry["width"], entry["height"])
        image.save(out_dir / f"label_{index+1:02d}.png")


def prediction_figures(pred_path, data_dir="data", out_dir="report/figures", coord="norm1000", count=6):
    entries = read_jsonl(Path(data_dir)/"splits"/"test.jsonl")
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(pred_path)}
    if {r["image_id"] for r in entries} != set(preds):
        raise ValueError("Figures require complete test predictions")
    ranked = []
    for e in entries:
        score = evaluate([e], {e["image_id"]: preds[e["image_id"]]}, coord, n_boot=40)
        m = score["micro"]
        ranked.append((m["f1"], m["fp"]+m["fn"], e["image_id"], e, m))
    ranked.sort(key=lambda row: (row[0], -row[1], row[2]))
    # Best two and worst four; if every prediction fails, captions report that honestly.
    candidates = list(reversed(ranked[-2:])) + ranked
    selected, seen = [], set()
    for row in candidates:
        if row[2] not in seen:
            selected.append(row)
            seen.add(row[2])
        if len(selected) == count:
            break
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    captions = ["# Ảnh minh họa dự đoán trên test\n",
                "Trái: ground truth (xanh). Phải: dự đoán (đỏ). Ghép cặp cùng lớp tại IoU ≥ 0,5.\n"]
    for i, (_, _, _, e, m) in enumerate(selected):
        with Image.open(Path(data_dir)/e["image_path"]) as source:
            left, right = source.convert("RGB"), source.convert("RGB")
        _draw_boxes(left, json.loads(objects_to_answer(e["objects"])), "lime", "norm1000", e["width"], e["height"])
        parsed = parse_output(preds[e["image_id"]])
        _draw_boxes(right, parsed or [], "red", coord, e["width"], e["height"])
        w, h = left.size
        canvas = Image.new("RGB", (2*w, h+70), "#101820")
        canvas.paste(left, (0, 36))
        canvas.paste(right, (w, 36))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), "Ground truth", fill="lime", font=_font(18))
        draw.text((w+12, 8), "Prediction", fill="red", font=_font(18))
        status = f"TP={m['tp']} FP={m['fp']} FN={m['fn']} F1={m['f1']:.3f}; JSON={'valid' if parsed is not None else 'invalid'}"
        draw.text((12, h+43), status, fill="white", font=_font(14))
        filename = f"prediction_{i+1:02d}.png"
        canvas.save(out_dir/filename)
        captions.append(f"## {e['image_id']}\n\n![{status}]({filename})\n\n{status}\n\n"
                        f"Raw output:\n```json\n{preds[e['image_id']]}\n```\n")
    (out_dir/"CAPTIONS.md").write_text("\n".join(captions), encoding="utf-8")
    return out_dir
