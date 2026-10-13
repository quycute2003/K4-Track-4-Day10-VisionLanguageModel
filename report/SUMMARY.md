# Kết quả đo từ các lần chạy thật

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | 0.0854 | [0.0607, 0.1117] | 1.0096 | 0.0905 | 0.0164 | 0.2160 |
| E1 | LoRA LLM | 33030144 | 10.12 | 46.91 | 0.2444 | [0.2105, 0.2814] | 1.4277 | 0.1659 | 0.1646 | 0.5403 |
| E2 | + projector | 142135296 | 8.17 | 47.85 | 0.3368 | [0.2989, 0.3762] | 1.1736 | 0.2149 | 0.2729 | 0.5891 |
| E3 | + LoRA ViT | 148426752 | 11.62 | 56.42 | 0.4122 | [0.3762, 0.4510] | 1.0482 | 0.2736 | 0.3333 | 0.6947 |

CI95 được bootstrap theo ảnh bởi công cụ chấm của đề. CI của từng F1 không phải CI của chênh lệch giữa hai mô hình; không kết luận ý nghĩa thống kê chỉ từ việc hai khoảng chồng lấp.

Chưa có evidence E4; bảng chỉ liệt kê các thí nghiệm đã được chấm. Xem nhật ký checkpoint để lấy số module, số token ảnh, tham số từng khối và loss. Xem REPORT.md để đọc phần phân tích và 10 câu trả lời.
