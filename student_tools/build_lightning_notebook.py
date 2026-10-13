"""Create a CPU-only notebook for exporting existing lab results on Lightning AI."""
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
# Day 10 — khôi phục và xuất E0–E3 trên Lightning AI / CPU

Bản này chỉ xử lý kết quả đã lưu; không huấn luyện, không suy luận VLM và không cần GPU.
Tải `lightning-ready.zip` vào cùng thư mục notebook trong Studio rồi chạy phần 0 và 9.
Gói được chuẩn bị cùng kết quả thật đã khôi phục từ Drive; phần E4/E5 được bỏ qua theo lựa chọn của bạn.
Nếu dùng bộ dữ liệu đầu vào khác, đặt `RESULTS_DIR` trỏ đến bản sao thư mục `day10-vlm` từ Google Drive.
""")
md("""
## 0. Mã nguồn và dữ liệu đã lưu — không dùng Google Colab

Không có `drive.mount`, Secrets, `files.upload` hoặc yêu cầu GPU. ZIP ở đường dẫn bên dưới
phải chứa mã và kết quả đã lưu. Thư mục làm việc dùng ổ đĩa Studio.
""")
code('''
import os, sys, zipfile, subprocess
from pathlib import Path

BASE_DIR = Path.cwd().resolve()
PACKAGE = BASE_DIR / "lightning-ready.zip"  # Đổi nếu ZIP ở thư mục khác
REPO_DIR = BASE_DIR / "day10-vlm"

if not (REPO_DIR / "lab_vlm/export_saved.py").exists():
    if not PACKAGE.exists():
        raise FileNotFoundError(f"Tải lightning-ready.zip lên Studio trước: {PACKAGE}")
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACKAGE) as archive:
        for member in archive.infolist():
            if not (REPO_DIR / member.filename).resolve().is_relative_to(REPO_DIR.resolve()):
                raise ValueError("Unsafe archive path")
        archive.extractall(REPO_DIR)

os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-export.txt"], check=True)

RESULTS_DIR = REPO_DIR  # Gói đã chứa report/ và các dự đoán dưới work/
DATA_DIR = REPO_DIR / "data"
EXPORT_DIR = REPO_DIR / "exports"
print("CPU export ready:", REPO_DIR)
''')
md("""
## 9. Khôi phục bảng, ảnh và ZIP kết quả

Đọc evidence/validation từ file, không phụ thuộc biến BEST hay run_module của phiên Colab cũ.
Nếu ảnh đã có, dùng lại. Nếu cần tạo ảnh, chỉ tải dataset công khai và xử lý trên CPU.
File ZIP được lưu trong `exports/`; tải bằng link dưới hoặc trong danh sách file của Studio.
REPORT.md vẫn cần họ tên và các câu trả lời cá nhân.
""")
code('''
from lab_vlm.export_saved import finish_saved_results
from IPython.display import display, FileLink, Markdown

export = finish_saved_results(
    repo_dir=REPO_DIR,
    results_dir=RESULTS_DIR,
    data_dir=DATA_DIR,
    output_dir=EXPORT_DIR,
)
display(Markdown((REPO_DIR / "report/SUMMARY.md").read_text(encoding="utf-8")))
display(FileLink(export.relative_to(REPO_DIR).as_posix()))

from scripts.check_submission import run_checks
for status, name, detail in run_checks(REPO_DIR):
    print(f"[{status}] {name}: {detail}")
print("Thiếu E4/E5 là lựa chọn đã chấp nhận; hoàn tất REPORT.md và lịch sử Git trước khi nộp.")
''')
for i, cell in enumerate(cells):
    cell["id"] = f"lightning-day10-{i:03d}"
nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(ROOT/"lab_day10_lightning.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print("Created CPU-only Lightning notebook")
