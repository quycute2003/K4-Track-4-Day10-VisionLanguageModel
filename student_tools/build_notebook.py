"""Rebuild the runnable Colab notebook from readable cell sources."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": (dedent(text).strip()+"\n").splitlines(True)})


md("""
# Lab Day 10 — Qwen3-VL-4B: bộ chạy Colab E0–E4

Notebook đã có mã thật trong `lab_vlm/`. Chọn **T4 GPU**, tải `colab-ready.zip`
ở ô đầu, rồi chạy từ trên xuống. Dữ liệu ở ổ tạm Colab; checkpoint, dự đoán và
bản sao báo cáo ở Drive. Chạy lại các ô sau khi mất phiên để tiếp tục từ checkpoint.

E4 mặc định so sánh **10% / 100% dữ liệu train**, chọn setup bằng **validation**.
E5 là bonus, mặc định tắt. Không cần tài khoản Langfuse. Đọc `COLAB_GUIDE.md` khi cần.
""")
md("""
## 0. Môi trường và mã nguồn (Colab T4)

Chọn `SOURCE = "upload"` để đưa bản mã vừa hoàn thiện lên Colab.
Nếu đã đẩy mã này lên GitHub, chọn `"github"`. Không dùng bản GitHub cũ còn notebook trống.
""")
code('''
import os, sys, subprocess, json, shutil, zipfile, hashlib
from pathlib import Path
from google.colab import files, drive

SOURCE = "upload"  # "upload" hoặc "github"
REPO_URL = "https://github.com/quycute2003/K4-Track-4-Day10-VisionLanguageModel.git"
REPO_DIR = Path("/content/day10-vlm")

if not (REPO_DIR / "lab_vlm" / "runtime.py").exists():
    if SOURCE == "upload":
        print("Chọn colab-ready.zip (bộ mã, chưa phải dataset).")
        uploaded = files.upload()
        packages = [name for name in uploaded if name.endswith(".zip")]
        if len(packages) != 1:
            raise ValueError("Hãy tải lên đúng một file ZIP chứa bộ mã.")
        REPO_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(packages[0]) as archive:
            for member in archive.infolist():
                target = (REPO_DIR / member.filename).resolve()
                if not target.is_relative_to(REPO_DIR.resolve()):
                    raise ValueError("Unsafe archive path")
            archive.extractall(REPO_DIR)
        del uploaded
    elif SOURCE == "github":
        subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
    else:
        raise ValueError("SOURCE phải là upload hoặc github")
if not (REPO_DIR / "lab_vlm" / "runtime.py").exists():
    raise FileNotFoundError("Bản mã này thiếu lab_vlm/. Hãy dùng colab-ready.zip mới.")
os.chdir(REPO_DIR)

# Install before importing Torch/Transformers/PEFT into this notebook session.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt", "pytest"], check=True)
from lab_vlm.environment import remove_incompatible_colab_torchao
remove_incompatible_colab_torchao()
subprocess.run(["nvidia-smi", "-L"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "student_checks"], check=True)
''')
code('''
drive.mount("/content/drive")
DRIVE_DIR = Path("/content/drive/MyDrive/day10-vlm")
WORK_DIR = DRIVE_DIR / "work"
WORK_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = REPO_DIR / "data"

# Controls shared across E1–E3 and E4 to preserve fair comparisons.
BATCH_SIZE = 2       # If T4 runs out of memory: 1
GRAD_ACCUM = 4       # When BATCH_SIZE=1, use 8 to keep effective batch size=8
MAX_STEPS = 300
INFER_BATCH_SIZE = 4 # Generation automatically retries at 2 or 1 on OOM
RUN_E5 = False      # Optional bonus; finish E0–E4 first
assert BATCH_SIZE * GRAD_ACCUM == 8, "Giữ effective batch size bằng 8"

if (DRIVE_DIR / "report").exists():
    shutil.copytree(DRIVE_DIR / "report", REPO_DIR / "report", dirs_exist_ok=True)
if (DRIVE_DIR / "REPORT.md").exists():
    shutil.copy2(DRIVE_DIR / "REPORT.md", REPO_DIR / "REPORT.md")

def run_module(module, *arguments):
    from lab_vlm.execution import run_module as stream_module
    stream_module(module, *arguments, cwd=REPO_DIR)

def backup_report():
    if (REPO_DIR / "report").exists():
        shutil.copytree(REPO_DIR / "report", DRIVE_DIR / "report", dirs_exist_ok=True)
    shutil.copy2(REPO_DIR / "REPORT.md", DRIVE_DIR / "REPORT.md")

import yaml
from huggingface_hub import HfApi
from lab_vlm.data import write_json

# Pin the same base model revision across sessions and all experiments.
revision_file = WORK_DIR / "model_revision.json"
if not revision_file.exists():
    write_json(revision_file, {"revision": HfApi().model_info("Qwen/Qwen3-VL-4B-Instruct").sha})
MODEL_REVISION = json.loads(revision_file.read_text())["revision"]

def config_for(number):
    cfg = yaml.safe_load((REPO_DIR / "configs" / f"e{number}.yaml").read_text())
    cfg.update(revision=MODEL_REVISION, coord="norm1000", max_new_tokens=256)
    if number in (1, 2, 3):
        cfg.update(batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM, max_steps=MAX_STEPS)
    return cfg

def store_config(cfg, slug):
    path = WORK_DIR / "configs" / f"{slug}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    return path

def latest(slug):
    from lab_vlm.runtime import checkpoint_path
    checkpoint = checkpoint_path(WORK_DIR / slug)
    if checkpoint is None:
        raise FileNotFoundError(f"Chưa có checkpoint cho {slug}")
    return checkpoint

def score_once(pred, slug, split="test", checkpoint=None, coord="norm1000", setup=None):
    folder = "evidence" if split == "test" else "validation"
    output = REPO_DIR / "report" / folder / f"{slug}.json"
    current_meta = json.loads((DATA_DIR / "splits" / "meta.json").read_text())
    pred_hash = hashlib.sha256(Path(pred).read_bytes()).hexdigest()
    if output.exists():
        saved = json.loads(output.read_text())
        if (saved["dataset_version"] != current_meta["dataset_version"] or saved["split"] != split
            or saved["coord"] != coord or saved["meta"].get("prediction_sha256") != pred_hash):
            raise ValueError(f"Bằng chứng cũ khác lần chạy này: {output}. Dùng thư mục Drive khác cho thí nghiệm mới.")
        backup_report()
        return saved
    arguments = ["score", "--pred", pred, "--name", slug.upper(), "--out", output,
                 "--data-dir", DATA_DIR, "--split", split, "--coord", coord]
    if checkpoint is not None:
        arguments.extend(["--checkpoint", checkpoint])
    if setup:
        arguments.extend(["--setup", setup])
    run_module("lab_vlm.cli", *arguments)
    backup_report()
    return json.loads(output.read_text())

print("Checkpoint và dự đoán:", WORK_DIR)
print("Base model revision:", MODEL_REVISION)
''')
md("""
## 1. Dữ liệu: tải và kiểm tra split theo phiên quay

Mặc định tải ZIP dataset công khai trực tiếp, không cần Secrets Kaggle.
Bạn cũng có thể đặt đường dẫn ZIP trên Drive, để trống để tải ZIP lên,
hoặc chọn `source` để dùng thư mục đã giải nén. `kaggle` là lựa chọn có Secrets.
""")
code('''
DATA_SOURCE = "zip"  # "zip", "kaggle", "source"
DATASET_ZIP = "https://www.kaggle.com/api/v1/datasets/download/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"
# Có thể thay URL bằng đường dẫn ZIP trên Drive; để trống sẽ mở hộp tải lên.
DATASET_SOURCE_DIR = "" # Thư mục chứa images/ và labels-YOLO/

if not (DATA_DIR / "splits" / "meta.json").exists():
    arguments = ["--out", str(DATA_DIR), "--work", "/content/day10-dataset"]
    if DATA_SOURCE == "kaggle":
        from google.colab import userdata
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
        arguments += ["--kaggle"]
    elif DATA_SOURCE == "zip":
        if not DATASET_ZIP:
            uploaded = files.upload()
            candidates = [name for name in uploaded if name.endswith(".zip")]
            if len(candidates) != 1:
                raise ValueError("Chọn đúng một dataset ZIP")
            DATASET_ZIP = str(REPO_DIR / candidates[0])
            del uploaded
        arguments += ["--zip", DATASET_ZIP]
    elif DATA_SOURCE == "source" and DATASET_SOURCE_DIR:
        arguments += ["--source", DATASET_SOURCE_DIR]
    else:
        raise ValueError("Kiểm tra DATA_SOURCE và đường dẫn dữ liệu")
    run_module("scripts.prepare_data", *arguments)

from lab_vlm.data import target_quality_report, verify_splits
splits, dataset_meta = verify_splits(DATA_DIR)
print("dataset_version:", dataset_meta["dataset_version"])
label_quality = {name: target_quality_report(entries) for name, entries in splits.items()}
write_json(REPO_DIR / "report/diagnostics/label_quality.json", label_quality)
for name, quality in label_quality.items():
    print(name, "zero-area labels omitted from targets:", quality["zero_area_annotations_skipped"])
from lab_vlm.reporting import ground_truth_preview
ground_truth_preview(splits["train"], DATA_DIR)
from IPython.display import display, Image as DisplayImage
for path in sorted((REPO_DIR / "report/figures/labels").glob("*.png")):
    display(DisplayImage(filename=str(path)))
backup_report()
''')
md("""
## 2. Nhãn: YOLO → JSON thang 0–1000

Hàm chuyển đổi kiểm tra làm tròn, kẹp tọa độ và thứ tự box. Nhãn gốc có chiều rộng/cao
bằng 0 được bỏ khỏi đáp án huấn luyện và ghi vào `report/diagnostics/label_quality.json`.
Box có diện tích dương bị co về 0 khi làm tròn được làm tròn ra ngoài riêng trên trục đó.
Ảnh xanh ở phần trên dùng JSON quy ngược về pixel, để kiểm tra bằng mắt.
""")
code('''
from lab_vlm.data import objects_to_answer, make_prompt
sample = splits["train"][0]
print(make_prompt())
print("Nhãn JSON:", objects_to_answer(sample["objects"]))
print("Nhãn pixel để đối chiếu:", objects_to_answer(sample["objects"], "pixel", sample["width"], sample["height"]))
''')
md("""
## 3. E0: zero-shot

Chạy đủ test và ghi output thô. Mỗi batch được lưu vào Drive ngay khi hoàn thành.
Chạy lại ô này sẽ tiếp tục những ảnh còn thiếu.
""")
code('''
from lab_vlm.cli import main
from scripts.labcommon import read_jsonl

# Recreate these variables after a fresh/restarted Colab session.
e0_cfg = store_config(config_for(0), "e0")
e0_pred = WORK_DIR / "e0_test.jsonl"

print("Bắt đầu E0...", flush=True)
main([
    "infer",
    "--config", str(e0_cfg),
    "--out", str(e0_pred),
    "--data-dir", str(DATA_DIR),
    "--batch-size", str(INFER_BATCH_SIZE),
])

e0_score = score_once(e0_pred, "e0", setup="zero-shot")
print("parse_valid_rate:", e0_score["parse_valid_rate"])
for row in read_jsonl(e0_pred)[:5]:
    print(row["image_id"], row["raw"], "\\n")
''')
md("""
## 4. Ví dụ huấn luyện và mask loss

`lab_vlm/encoding.py` kiểm tra prompt là tiền tố, mask bằng -100 và nối tensor ảnh.
Trước mỗi lần train, chương trình in đáp án được tính loss và xác nhận **220 token ảnh**.
Các kiểm tra riêng ở `student_checks/` đã chạy trong bước 0.
""")
code('''
import inspect
from lab_vlm.encoding import encode_example, collate
print(inspect.getsource(encode_example))
print(inspect.getsource(collate))
''')
md("""
## 5. Setup LoRA / đóng băng E1, E2, E3

E1: LoRA LLM; E2: thêm projector đầy đủ; E3: thêm LoRA ViT.
Mỗi run xác nhận số module 252/252/348 và ghi số tham số thực tế.
ViT và lm_head giữ ngoài 4-bit; mọi tham số được cập nhật ở float32, tính toán với autocast fp16.
""")
code('''
config_paths = {f"e{number}": store_config(config_for(number), f"e{number}") for number in (1, 2, 3)}
for slug, path in config_paths.items():
    print(slug, yaml.safe_load(path.read_text()))
''')
md("""
## 6. Huấn luyện E1–E3

Mỗi run bắt đầu từ cùng model gốc, không nối tiếp adapter của run trước.
Checkpoint được ghi sau mỗi 100 bước và bước cuối, gồm adapter, projector,
optimizer, lịch LR và trạng thái ngẫu nhiên. Phiên bị ngắt sẽ quay lại checkpoint hoàn chỉnh gần nhất.
""")
code('''
for slug in ("e1", "e2", "e3"):
    run_module("lab_vlm.cli", "train", "--config", config_paths[slug],
               "--data-dir", DATA_DIR, "--run-dir", WORK_DIR / slug)
    manifest = json.loads((latest(slug) / "manifest.json").read_text())
    print(slug, {key: manifest[key] for key in ("trainable_params", "peak_vram_gb", "train_minutes")})
''')
md("""
## 7. Suy luận và chấm E1–E3

Chấm validation riêng để lựa chọn setup cho E4. Chấm test đủ ảnh và giữ kết quả đã tạo.
Nếu cần gỡ lỗi 40 ảnh validation, dùng CLI `infer --split val --limit 40` với tên output khác.
""")
code('''
validation = {}
for slug in ("e1", "e2", "e3"):
    for split in ("val", "test"):
        pred = WORK_DIR / f"{slug}_{split}.jsonl"
        run_module("lab_vlm.cli", "infer", "--config", config_paths[slug], "--checkpoint", latest(slug),
                   "--out", pred, "--data-dir", DATA_DIR, "--split", split, "--batch-size", INFER_BATCH_SIZE)
        measured = score_once(pred, slug, split, latest(slug), setup=slug.upper())
        if split == "val":
            validation[slug] = measured["micro"]["f1"]
BEST = max(validation, key=validation.get)
print("F1 validation:", validation, "→ Chọn", BEST)
write_json(WORK_DIR / "e4_selection.json", {"selected": BEST, "criterion": "validation micro F1", "scores": validation})
''')
md("""
## 8. E4 (ablation dữ liệu) và E5 (GRPO tùy chọn)

E4 đổi **đúng train_fraction**: 10% và 100%; các tham số khác giữ theo setup chọn bằng validation.
Run 100% dùng lại checkpoint và dự đoán baseline, không huấn luyện lại.
""")
code('''
baseline = json.loads((latest(BEST) / "manifest.json").read_text())["config"]
e4_cfg = dict(baseline, name="E4_DATA_SIZE_0.1", train_fraction=0.1)
e4_path = store_config(e4_cfg, "e4_data_size_0.1")
run_module("lab_vlm.cli", "train", "--config", e4_path, "--data-dir", DATA_DIR,
           "--run-dir", WORK_DIR / "e4_data_size_0.1")
e4_pred = WORK_DIR / "e4_data_size_0.1_test.jsonl"
run_module("lab_vlm.cli", "infer", "--config", e4_path, "--checkpoint", latest("e4_data_size_0.1"),
           "--out", e4_pred, "--data-dir", DATA_DIR, "--batch-size", INFER_BATCH_SIZE)
score_once(e4_pred, "e4_data_size_0.1", checkpoint=latest("e4_data_size_0.1"), setup=f"{BEST.upper()} train=10%")
score_once(WORK_DIR / f"{BEST}_test.jsonl", "e4_data_size_1.0", checkpoint=latest(BEST), setup=f"{BEST.upper()} train=100%")
''')
md("""
E5 dùng recipe TRL 1.9.2, khởi tạo từ baseline tốt nhất, tiếp tục LoRA và giữ projector SFT.
Reward đã có test: hoàn hảo, JSON hỏng, sai lớp, thừa box, rỗng và box khổng lồ.
Recipe này cần kiểm chứng trên Colab; T4 có thể cần điều chỉnh bộ nhớ cho GRPO.
""")
code('''
if RUN_E5:
    run_module("lab_vlm.cli", "grpo", "--checkpoint", latest(BEST), "--run-dir", WORK_DIR / "e5", "--data-dir", DATA_DIR)
    e5_config = store_config(json.loads((latest("e5") / "manifest.json").read_text())["config"], "e5")
    e5_pred = WORK_DIR / "e5_test.jsonl"
    run_module("lab_vlm.cli", "infer", "--config", e5_config, "--checkpoint", latest("e5"),
               "--out", e5_pred, "--data-dir", DATA_DIR, "--batch-size", INFER_BATCH_SIZE)
    score_once(e5_pred, "e5", checkpoint=latest("e5"), setup="GRPO from "+BEST.upper())
else:
    print("Bỏ qua E5 (bonus).")
''')
md("""
## 9. Báo cáo, ảnh minh họa và xuất kết quả

Ô này điền **bảng số liệu** vào REPORT.md, tạo ≥6 ảnh ground truth/dự đoán và bản tổng hợp.
10 câu trả lời và họ tên vẫn cần bạn viết theo kết quả quan sát của mình.
Kiểm tra thang tọa độ sai trên E0 dùng một evidence phụ, giữ nguyên kết quả chính.
""")
code('''
# Restore state needed for export after restarting Colab. Run sections 0 and 1
# first to recover source code, Drive access, reports and local dataset images.
import json
from IPython.display import display, Image as DisplayImage

saved_validation = {}
for slug in ("e1", "e2", "e3"):
    validation_path = REPO_DIR / "report/validation" / f"{slug}.json"
    if validation_path.exists():
        measured = json.loads(validation_path.read_text(encoding="utf-8"))
        if measured["split"] != "val":
            raise ValueError(f"Expected validation evidence: {validation_path}")
        saved_validation[slug] = measured["micro"]["f1"]
if not saved_validation:
    raise FileNotFoundError("Chưa có kết quả validation đã lưu. Kiểm tra Drive/report/validation hoặc hoàn tất phần 7.")
BEST = max(saved_validation, key=saved_validation.get)
e0_pred = WORK_DIR / "e0_test.jsonl"
e0_score_path = REPO_DIR / "report/evidence/e0.json"
e0_score = json.loads(e0_score_path.read_text(encoding="utf-8"))
print("Khôi phục model minh họa từ validation:", BEST.upper())

run_module("lab_vlm.cli", "report", "--fill-table")
run_module("lab_vlm.cli", "figures", "--pred", WORK_DIR / f"{BEST}_test.jsonl", "--data-dir", DATA_DIR)
coord_check = REPO_DIR / "report/diagnostics/e0_wrong_coord.json"
run_module("lab_vlm.cli", "score", "--pred", e0_pred, "--name", "E0_WRONG_COORD",
           "--out", coord_check, "--data-dir", DATA_DIR, "--coord", "pixel")
print("E0 chấm đúng:", e0_score["micro"]["f1"], "E0 cố ý chấm sai thang:", json.loads(coord_check.read_text())["micro"]["f1"])
for image_path in sorted((REPO_DIR / "report/figures").glob("prediction_*.png")):
    display(DisplayImage(filename=str(image_path)))
backup_report()

from lab_vlm.packaging import package_project
export = package_project(REPO_DIR, DRIVE_DIR / "day10-results.zip", include_report=True)
print("Kết quả đã lưu trên Drive:", export)
files.download(str(export))

# Expected to report missing personal answers/commit history until the final submission is ready.
result = subprocess.run([sys.executable, "-m", "scripts.check_submission"], cwd=REPO_DIR)
print("Điền REPORT.md, đưa mã và report/ về repo GitHub, tạo ≥5 commit thực chất rồi kiểm tra lại.")
''')

notebook = {"cells": cells, "metadata": {
    "accelerator": "GPU", "colab": {"name": "lab_day10.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells):
    cell["id"] = f"day10-{index:03d}"
(ROOT / "lab_day10.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(f"Wrote {len(cells)} notebook cells")
