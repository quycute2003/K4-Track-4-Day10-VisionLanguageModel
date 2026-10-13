# Chạy bộ mã Day 10 trên Colab

## 1. Đưa mã lên Colab

1. Tải `lab_day10.ipynb` lên [Google Colab](https://colab.research.google.com/).
2. Chọn **Runtime → Change runtime type → T4 GPU**.
3. Giữ `SOURCE = "upload"` trong phần 0. Chạy ô đầu và chọn **colab-ready.zip** khi hộp tải lên xuất hiện. ZIP này chứa mã hoàn thiện, không chứa dữ liệu hoặc trọng số.
4. Chạy ô tiếp theo để kết nối Google Drive. Kết quả mặc định ở `MyDrive/day10-vlm/`.
5. Nếu đã đẩy mã mới lên repo GitHub, có thể chọn `SOURCE = "github"` để clone repo thay cho tải ZIP. Đừng clone bản repo cũ còn notebook khung.

Có thể chọn **Run all** sau khi đặt các tùy chọn. Các hộp tải file, quyền Drive và Secrets vẫn cần bạn thao tác. Mỗi ô có thể chạy riêng để xem lỗi hoặc tiếp tục sau khi mất phiên.

## 2. Chọn nguồn dữ liệu

Trong phần 1, chọn một cách:

- `DATA_SOURCE = "zip"` **(mặc định)**: `DATASET_ZIP` đã chứa URL tải công khai của dataset, không cần khóa Kaggle. Có thể thay bằng đường dẫn ZIP trên Drive, hoặc để trống để tải ZIP dataset lên. Đây là ZIP dữ liệu, khác `colab-ready.zip`.
- `DATA_SOURCE = "kaggle"`: tạo Secrets `KAGGLE_USERNAME` và `KAGGLE_KEY`, bật quyền truy cập cho notebook. Không đưa khóa vào mã hay Git.
- `DATA_SOURCE = "source"`: đặt `DATASET_SOURCE_DIR` thành thư mục có `images/` và `labels-YOLO/`.

Notebook dùng nguyên công cụ chia dữ liệu của đề, kiểm tra không giao phiên quay hoặc image ID giữa các split, và vẽ 6 ảnh có nhãn.

Dataset gốc có một nhãn crack trong `vlcsnap-00058.txt` với chiều cao YOLO bằng 0 (`0.639062 0.786111 0.003125 0.000000`). Nhãn này không thể tạo box có diện tích hợp lệ. Mã bỏ riêng nhãn đó khi tạo đáp án SFT/GRPO, vẫn giữ ảnh, nhãn gốc và công cụ chấm nguyên vẹn. Chi tiết được lưu vào `report/diagnostics/label_quality.json` và `work/<run>/target_quality.json`; ghi cách xử lý này trong phần nhìn lại của báo cáo. Với box có diện tích dương nhưng bị co về 0 khi làm tròn, mã dùng floor/ceil riêng trên trục bị co, giữ nguyên cách làm tròn cho box bình thường.

Nếu đang dùng bản notebook cũ và gặp `SecretNotFoundError`, đổi hai dòng đầu ô dữ liệu thành:

```python
DATA_SOURCE = "zip"
DATASET_ZIP = "https://www.kaggle.com/api/v1/datasets/download/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"
```

Chạy lại ô dữ liệu rồi tiếp tục các phần sau. URL đã được kiểm tra trả HTTP 200 và đúng định dạng ZIP khi sửa lỗi này. Kaggle hỗ trợ [tải dataset công khai không cần xác thực](https://www.kaggle.com/product-announcements/485439).

## 3. Chạy thí nghiệm

| Phần | Việc thực hiện |
|---|---|
| 0–2 | Cài môi trường, lấy dữ liệu, kiểm tra nhãn |
| 3 | E0 zero-shot trên toàn bộ test; in 5 output thô |
| 4–5 | Xem mã mask loss và cấu hình LoRA |
| 6 | Train E1, E2, E3, mỗi run từ cùng model gốc |
| 7 | Suy luận validation và test; chọn baseline bằng F1 validation |
| 8 | E4: 10%/100% dữ liệu train; 100% dùng lại baseline |
| 9 | Điền bảng số, tạo 6 ảnh có chú thích, xuất ZIP kết quả |

Model gốc được cố định theo revision trên Hugging Face, lưu ở Drive để các phiên tiếp theo dùng cùng trọng số. Tất cả E1–E3 dùng cùng batch, số bước và seed.

E4 mặc định đổi đúng một biến là **tỷ lệ dữ liệu train**. Subset 10% được chọn bằng shuffle với seed cố định; vẫn chạy 300 bước như baseline. Vì số bước giữ nguyên, thí nghiệm đo hiệu quả kích thước dữ liệu với cùng số lần cập nhật, không cùng số epoch.

E5 có recipe GRPO và test reward nhưng là **bonus, mặc định tắt**. Đặt `RUN_E5 = True` để chạy sau E0–E4. Recipe dựa trên [tài liệu TRL 1.9.2](https://huggingface.co/docs/trl/v1.9.2/en/grpo_trainer), chưa được chạy full model trên T4 trong quá trình chuẩn bị mã này.

## 4. Lưu và tiếp tục

- Checkpoint, log loss và dự đoán: `MyDrive/day10-vlm/work/`.
- Bằng chứng chấm: `MyDrive/day10-vlm/report/evidence/`.
- Bảng tổng hợp: `MyDrive/day10-vlm/report/SUMMARY.md`.
- Ảnh: `MyDrive/day10-vlm/report/figures/`, kèm `CAPTIONS.md` và output thô.
- Báo cáo và ZIP kết quả: `MyDrive/day10-vlm/REPORT.md`, `day10-results.zip`.

SFT lưu checkpoint mỗi 100 bước và ở bước cuối. Checkpoint gồm adapter, projector đã huấn luyện của E2/E3, optimizer, scheduler, GradScaler và RNG. Mất phiên thì mở notebook, làm lại phần 0–2 và chạy lại phần bị ngắt. Một phần bước chưa được checkpoint sẽ phải chạy lại.

Nếu đã chấm xong E0–E3 và chỉ cần xuất kết quả sau khi mất runtime: chạy lại đủ các ô phần **0 và 1**, rồi sang **9**; không chạy lại train hoặc suy luận ở phần 3–8. Phần 0 khôi phục mã, kết nối Drive và sao chép report đã lưu; phần 1 tải lại ảnh/split ở ổ tạm để vẽ ảnh và chấm kiểm tra thang tọa độ. Ô 9 mới tự đọc lại `BEST` từ `report/validation/*.json`, `e0_score` từ `report/evidence/e0.json` và `e0_pred` từ Drive. Nếu validation/evidence chưa được lưu, cần hoàn tất phần chấm ở phần 7 bằng checkpoint sẵn có.

Suy luận ghi kết quả theo batch; chạy lại sẽ bỏ qua ảnh đã có. Nếu dòng cuối bị cắt do mất phiên, chương trình bỏ dòng đó và suy luận lại ảnh tương ứng. Công cụ từ chối ghép kết quả khác model/dữ liệu/thang tọa độ.

Notebook hiển thị stdout/stderr của chương trình con trực tiếp. E0 in giai đoạn nạp model, số ảnh đã lưu, batch đang xử lý và thời gian còn lại ước tính. Khi nạp model hoặc sinh kết quả kéo dài, một dòng trạng thái được in mỗi 30 giây; dòng này chỉ cho biết tác vụ chưa hoàn thành, không bảo đảm GPU đang tiến triển.

Nếu bạn đổi cấu hình sau khi đã chạy, dùng **DRIVE_DIR mới** để giữ bằng chứng cũ. Đừng trộn checkpoint hoặc evidence giữa hai cấu hình.

## 5. Khi hết bộ nhớ

Suy luận tự giảm batch từ 4 xuống 2 hoặc 1. Huấn luyện dừng khi hết bộ nhớ hoặc gặp loss/gradient không hữu hạn.

Nếu T4 báo hết bộ nhớ khi train, đổi `BATCH_SIZE = 1`, `GRAD_ACCUM = 8` và dùng thư mục Drive mới. Áp dụng cùng thay đổi cho E1–E3 và E4 để giữ so sánh công bằng. Notebook dùng SDPA, gradient checkpointing và NF4; ViT và `lm_head` không lượng tử hóa theo yêu cầu đề.

Mã chạy dùng thêm tên đầy đủ `model.visual` trong danh sách loại trừ lượng tử hóa. Bài thử với Transformers 5.19.0 cho thấy chỉ `visual` như pseudo code chưa loại được các lớp con `model.visual.*`. Chương trình kiểm tra sau khi nạp và dừng nếu ViT hoặc lm_head vẫn bị lượng tử hóa.

Nếu E3 báo `torchao 0.10.0` không tương thích, gỡ package tùy chọn này bằng `pip uninstall -y torchao` trong phiên Colab rồi xóa cache của `peft.import_utils.is_torchao_available`. Bài dùng BitsAndBytes, không cần TorchAO. [PEFT 0.21.2 yêu cầu TorchAO ≥0.16.0 khi package có mặt](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/import_utils.py). Notebook mới tự gỡ riêng các bản TorchAO không tương thích ở phần 0; mã E3 cũng kiểm tra trước khi nạp model. Chạy lại riêng E3 bằng config cũ và `work/e3`; E1/E2 đã lưu không cần huấn luyện lại.

## 6. Hoàn tất báo cáo và nộp

Notebook tự điền **bảng kết quả và dataset_version** vào `REPORT.md`; giữ nguyên họ tên và 10 câu cần tự trả lời. Không tạo số liệu trước khi chạy, không điền kết luận thực nghiệm bằng giả định.

1. Lấy ZIP kết quả trên Drive và chép mã cùng `report/`, `REPORT.md` về repo.
2. Xem output thô và ảnh; điền họ tên, 10 câu và phần nhìn lại. CI95 của từng F1 không phải CI95 của chênh lệch hai model.
3. Đảm bảo repo công khai và có ≥5 commit thể hiện tiến trình thực tế.
4. Trên Windows, kiểm tra bằng `py -m pytest -q tests student_checks` và `py -m scripts.check_submission`.
5. Không đưa `data/`, `work/`, khóa hoặc trọng số vào Git. `scripts/` và `tests/` của đề được giữ nguyên.

Nếu chạy từ ZIP, Colab không có lịch sử Git của repo; mục Git của checker có thể WARN. Báo cáo chưa trả lời sẽ FAIL đúng như đề yêu cầu. Hãy kiểm tra lần cuối trong repo thật trên máy sau khi hoàn tất báo cáo.

## Cấu trúc mã bổ sung

- `lab_vlm/data.py`: nhãn, prompt, subset, lịch batch, reward.
- `lab_vlm/encoding.py`: chat template, mask loss, collate.
- `lab_vlm/runtime.py`: model, LoRA, SFT, checkpoint, suy luận.
- `lab_vlm/grpo.py`: E5 tùy chọn.
- `lab_vlm/reporting.py`: bảng và ảnh từ kết quả thật.
- `lab_vlm/cli.py`: lệnh chạy chung.
- `student_checks/`: kiểm tra mã bổ sung, tách khỏi tests của đề.

Các API được đối chiếu với [Qwen3-VL trong Transformers](https://huggingface.co/docs/transformers/model_doc/qwen3_vl) và [hướng dẫn quantization của PEFT](https://huggingface.co/docs/peft/developer_guides/quantization). Không cần kết nối dịch vụ theo dõi bên ngoài để chạy lab.

## Kiểm tra đã thực hiện trước khi bàn giao

- 87 kiểm tra đạt, 1 kiểm tra của đề được bỏ qua; không thay đổi `scripts/` hoặc `tests/`.
- Notebook hợp lệ về cấu trúc và cú pháp tất cả ô mã.
- Với đúng Transformers 5.19.0 / PEFT 0.21.2: xác nhận 252 module LLM, 96 module ViT, 109.105.152 tham số projector và 220 token cho ảnh 640×360 bằng config/processor thật.
- Với BitsAndBytes 0.50.2 trên GPU RTX 3050 4 GB: mô hình Qwen3-VL thu nhỏ với trọng số ngẫu nhiên chạy được NF4, cập nhật LoRA/projector, lưu và nạp checkpoint, suy luận batch 2. Sau mô phỏng ngắt phiên, trọng số adapter/projector khớp lần chạy liên tục.
- TRL 1.9.2: import trainer và khởi tạo cấu hình E5 thành công; reward có kiểm tra riêng.
- Rà đủ 2.009 file nhãn YOLO / 4.737 box của dataset thật: có đúng một nhãn chiều cao 0. Sau xử lý, cả thang 0–1000 lẫn pixel đều tạo được JSON box hợp lệ cho toàn bộ file.

Các bài thử dùng dữ liệu tổng hợp chỉ nằm trong `work/`, không tạo evidence của bài lab. **Chưa chạy huấn luyện model 4B đầy đủ hoặc GRPO trên dữ liệu mặt đường/T4**. Kết quả F1, VRAM và thời gian thực tế sẽ được sinh khi bạn chạy notebook.
