# Xuất kết quả trên Lightning AI khi Colab hết GPU

Bản `lab_day10_lightning.ipynb` chỉ có phần **0 và 9**, chạy bằng CPU. Các run E0–E3 đã được khôi phục từ Drive và kiểm tra lại bằng dự đoán gốc/công cụ chấm. Không cần tải model 4B, không train hoặc suy luận lại, và không cần gói TorchAO.

## Dùng bộ đã chuẩn bị

1. Tạo/mở một Studio trên Lightning AI, dùng CPU cho phần xuất kết quả.
2. Tải `lightning-ready.zip` và `lab_day10_lightning.ipynb` vào cùng thư mục Studio.
3. Mở notebook, chạy phần **0**, rồi phần **9**. Nếu ZIP ở vị trí khác, sửa biến `PACKAGE` trong ô 0.
4. Tải `day10-vlm/exports/day10-results.zip` từ link notebook hoặc danh sách file Studio.

Lightning hỗ trợ lưu file bền vững và tải file/thư mục vào Studio; xem [tài liệu Studio](https://lightning.ai/docs/platform/build/ai-studio) và [Lightning Drive](https://lightning.ai/docs/platform/team-management/teamspaces/drive).

`lightning-ready.zip` là gói **để khôi phục**, gồm mã, report và các file dự đoán nhỏ trong `work/`. `day10-results.zip` là gói **để chép vào repo nộp**, gồm mã và report, loại bỏ `work/`, `data/`, trọng số và khóa. Không commit thư mục `work/` hoặc `exports/`.

## Khác với Colab

- Phần 0 dùng đường dẫn Studio, không dùng `/content/drive`, `drive.mount`, Secrets hay `files.upload`.
- Chỉ cài `requirements-export.txt` (Pillow và PyYAML), không cài thư viện GPU.
- Phần 9 tự đọc các kết quả đã lưu; không dùng biến `BEST`, `e0_score` hoặc `run_module` còn sót từ phiên trước.
- ZIP được lưu thành file thường và hiện link tải, thay cho `google.colab.files.download`.
- Nếu đã có 6 ảnh dự đoán và kiểm tra thang tọa độ, dùng lại. Nếu chưa có, tải dataset công khai để vẽ/chấm trên CPU, không nạp VLM.

## Nếu dùng bản sao Drive riêng

Tải thư mục `MyDrive/day10-vlm` từ Drive về máy và đưa vào Studio. Đặt `RESULTS_DIR` trỏ tới thư mục đã giải nén, có cấu trúc:

```
day10-vlm/
  REPORT.md
  report/evidence/e0.json ... e3.json
  report/validation/e1.json ... e3.json
  work/e0_test.jsonl ... e3_test.jsonl
  work/e1/run.json ... e3/run.json  # số liệu và cấu hình, không phải trọng số
```

Chỉ lấy report và các file nhỏ trên là đủ; không cần chuyển các thư mục checkpoint. Nếu evidence hoặc dự đoán chưa hoàn thành, xuất báo cáo không thể tạo thay số liệu đó: cần tiếp tục suy luận/chấm từ adapter đã lưu trên một máy có GPU. Notebook Lightning này dành riêng cho bước xuất, không thay thế phần train/infer.

Theo lựa chọn hiện tại, bỏ E4 và E5. Evidence E4 không được tạo giả. REPORT.md đã hoàn thành với họ tên, mã học viên, số liệu và 10 câu trả lời; bước đóng gói giữ nguyên các câu trả lời này.

## Kiểm tra trước khi bàn giao

Notebook được kiểm tra cấu trúc/cú pháp, pipeline xuất chạy được trong phiên mới với package đã giải nén và không import Torch. Dự đoán E0–E3 có đủ 311 ảnh mỗi run, hash khớp evidence đã lưu; F1/CI được đối chiếu lại bằng công cụ chấm gốc trên đúng dataset version. Đây là kiểm tra trên máy hiện tại, chưa phải lần chạy trong một Studio Lightning thật. Kiểm tra gốc đòi dấu điền mẫu trong REPORT.md không còn phù hợp khi báo cáo đã hoàn thành; giữ nguyên kiểm tra đó và bỏ chọn khi chạy bộ kiểm tra chức năng, theo lệnh trong README.
