# HƯỚNG DẪN CHẠY DEMO LƯỢNG TỬ HÓA TUYẾN TÍNH (LINEAR QUANTIZATION PTQ INT8)

## 1. Cài đặt môi trường
Đảm bảo đã cài đặt `tensorflow` và `numpy`:
```bash
pip install tensorflow numpy
```

## 2. Thực thi script
Chạy trực tiếp file script kiểm thử:
```bash
cd AIOT
python quantization_demo.py
```

## 3. Đầu ra thu được:
1. `model_fp32.tflite`: Mô hình TFLite ở định dạng dấu phẩy động 32-bit nguyên bản.
2. `model_int8.tflite`: Mô hình TFLite Integer-only đã lượng tử hóa toàn bộ sang INT8.
3. Bảng báo cáo benchmark chi tiết in ra terminal:
   - Dung lượng (Model Size) FP32 vs INT8 (chứng minh giảm ~75%).
   - Độ trễ suy luận trung bình (Average Latency ms/sample) và Throughput (FPS).
   - Đo lường sai số: Mean Absolute Error (MAE), Mean Squared Error (MSE), Max Difference, Cosine Similarity và tỷ lệ khớp nhãn Top-1.
