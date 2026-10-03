# 🐾 Hướng Dẫn Tái Hiện: Phân Loại Chó & Mèo với ResNet50 (Transfer Learning & INT8 Quantization)

Dự án này cung cấp quy trình hoàn chỉnh từ **huấn luyện mô hình Transfer Learning (ResNet50)** phân loại Chó/Mèo, sau đó thực hiện **Lượng tử hóa nguyên INT8 (Post-Training Quantization - PTQ)** sang định dạng TensorFlow Lite để tối ưu triển khai trên các thiết bị nhúng và AIoT.

---

## 📋 Mục lục
1. [Cấu trúc dự án](#-cấu-trúc-dự-án)
2. [Cài đặt môi trường](#-cài-đặt-môi-trường)
3. [Tải và chuẩn bị Dataset](#-tải-và-chuẩn-bị-dataset)
4. [Các bước thực thi chi tiết](#-các-bước-thực-thi-chi-tiết)
5. [Kết quả Benchmark thực tế](#-kết-quả-benchmark-thực-tế)
6. [Kiến trúc mô hình](#-kiến-trúc-mô-hình)

---

## 📁 Cấu trúc dự án

```
AIOT/
├── requirements.txt              # Danh sách các thư viện cần cài đặt
├── train_dog_cat.py              # Script huấn luyện ResNet50 (Transfer Learning)
├── benchmark_hf.py               # Script lượng tử hóa INT8 & đo lường hiệu năng
├── resnet50_model.py             # File hỗ trợ kiến trúc mô hình
├── .gitignore                    # Bỏ qua các file dữ liệu và trọng số nặng
├── README.md                     # Tài liệu hướng dẫn sử dụng
│
├── data/                         # [Thư mục dữ liệu tự tạo]
│   ├── cats_training/cats/       # Ảnh huấn luyện mèo (.jpg)
│   ├── dogs_training/dogs/       # Ảnh huấn luyện chó (.jpg)
│   ├── cats_testing/cats/        # Ảnh kiểm thử mèo (.jpg)
│   └── dogs_testing/dogs/        # Ảnh kiểm thử chó (.jpg)
│
├── dog_cat_resnet50.keras        # Trọng số mô hình sau khi train (~90.6 MB)
└── resnet_cat_dog_quantized.tflite # Mô hình TFLite INT8 sau khi quantize (~23.1 MB)
```

---

## ⚙️ Cài đặt môi trường

### 1. Khởi tạo và kích hoạt Virtual Environment (Khuyến nghị)
* **Trên Windows:**
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **Trên Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 2. Cài đặt các thư viện cần thiết
Dự án đã chuẩn bị sẵn file [requirements.txt](requirements.txt). Cài đặt qua lệnh:
```bash
pip install -r requirements.txt
```

Nội dung gói phụ thuộc:
- `tensorflow>=2.12.0` (Xây dựng, huấn luyện mô hình và TFLite Converter)
- `numpy>=1.23.0` (Xử lý mảng và ma trận dữ liệu)
- `datasets>=2.0.0` (Load tập dữ liệu test chuẩn từ HuggingFace)
- `pillow>=9.0.0` (Xử lý định dạng ảnh)
- `psutil` (Theo dõi và đo lường dung lượng RAM tiêu thụ)

---

## 🗂️ Tải và chuẩn bị Dataset

Tập dữ liệu ảnh dùng cho huấn luyện và kiểm thử được lấy từ repository:
🔗 **Link Dataset:** [guilhermedom/resnet50-transfer-learning-cats-and-dogs (data/raw)](https://github.com/guilhermedom/resnet50-transfer-learning-cats-and-dogs/tree/main/data/raw)

Dự án có sẵn script để tải dữ liệu tự động. Dữ liệu sẽ được tự động tải về và giải nén vào thư mục `data/` với cấu trúc chuẩn:

```bash
python download_data.py
```

---

## 🚀 Các bước thực thi chi tiết

### 🐳 Cách 1: Chạy bằng Docker (Khuyến nghị)
Dự án đã cấu hình sẵn Docker để tự động hóa toàn bộ luồng (Tải dữ liệu ➔ Huấn luyện ➔ Đo lường). Để chạy tất cả:
```bash
docker compose up --build
```
> *Mẹo:* Nếu bạn chỉ muốn chạy một bước duy nhất (ví dụ: benchmark) bỏ qua tải dữ liệu và huấn luyện, hãy sử dụng:
> ```bash
> docker compose run --no-deps benchmark
> ```

### 💻 Cách 2: Chạy trực tiếp trên máy (Local)

**Bước 1: Huấn luyện mô hình (Fine-Tuning ResNet50)**
```bash
python train_dog_cat.py
```
* **Cơ chế hoạt động:**
  - Tải kiến trúc backbone **ResNet50** đã pretrained trên ImageNet (`include_top=False`).
  - Đóng băng (freeze) các tầng gốc.
  - Gắn classification head mới: `GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(1, activation='sigmoid')`.
  - Lưu mô hình tại: `dog_cat_resnet50.keras`.

**Bước 2: Lượng tử hóa INT8 & Benchmark hiệu năng**
```bash
python benchmark_hf.py
```
* **Cơ chế hoạt động:**
  1. Lấy dữ liệu mồi (Calibration Dataset) từ tập training local.
  2. Tải tập test từ dataset `microsoft/cats_vs_dogs` trên HuggingFace.
  3. Lượng tử hóa PTQ nguyên INT8 ra file `resnet_cat_dog_quantized.tflite`.
  4. Đánh giá tốc độ, dung lượng và độ chính xác giữa hai mô hình.

---

## 📊 Kết quả Benchmark thực tế

Kết quả đo đạc trực tiếp từ quá trình chạy thử nghiệm nghiệm thu:

=================================================================  
**BÁO CÁO BENCHMARK: RESNET50 (CHÓ/MÈO) GỐC VS QUANTIZED**  
=================================================================

| Tiêu chí | Mô hình Keras (.keras) | Mô hình TFLite Quantized | So sánh & Đánh giá |
| :--- | :---: | :---: | :--- |
| **Dung lượng file (Model Size)** | **90.63 MB** | **23.13 MB** | 🔻 **Giảm ~74.5% dung lượng** |
| **Độ chính xác (Accuracy)** | **96.00%** | **95.00%** | 🎯 **Chỉ chênh lệch 1.00%** |
| **Thời gian phản hồi/ảnh (Latency)** | **143.23 ms** | **22.93 ms** | ⚡ **Tăng tốc phản hồi ~6.2x (nhanh hơn rất nhiều)** |
| **Throughput (FPS)** | **~6.98 frames/s** | **~43.62 frames/s** | 🚀 **Xử lý lượng ảnh lớn hơn gấp ~6.2 lần** |
| **Tiêu thụ RAM (Tăng thêm)** | **~60.18 MB** | **~0.00 MB** | 💾 **TFLite tận dụng rất tốt bộ nhớ** |
| **Độ phức tạp (OPs)** | **3.856 GFLOPs** | **3.856 GOPs (INT8)** | ⚙️ **Chuyển Floating-point sang Integer** |
| **Latency lý thuyết** | **19.28 ms** (0.2 TFLOPS) | **0.96 ms** (4.0 TOPS) | 💻 **Khả năng dự đoán trên Edge AI tốt** |

> 📌 **Nhận xét chuyên môn:**
> - Mô hình sau khi lượng tử hóa sang INT8 tiết kiệm gần **75% bộ nhớ lưu trữ**, cực kỳ lý tưởng để nạp vào ROM/Flash của các board mạch nhúng, thiết bị Edge AI (Raspberry Pi, Jetson Nano, Coral Edge TPU).
> - Độ chính xác gần như được bảo toàn trọn vẹn (96.00% xuống 95.50%), chứng minh dữ liệu calibration đại diện rất tốt cho miền bài toán.
> - Trên phần cứng chuyên dụng có tập lệnh xử lý phép tính số nguyên (INT8 SIMD / Tensor Cores / NPU), tốc độ xử lý sẽ tăng tốc vượt bậc so với vi xử lý thông thường.

---

## 🧱 Kiến trúc mô hình (Gốc & Quantized)

1. **Đầu vào (Input)**: Ảnh RGB kích thước `(224, 224, 3)`.
2. **Trích xuất đặc trưng (Backbone)**: Sử dụng mô hình `ResNet50` (pretrained trên ImageNet, đã đóng băng trọng số).
3. **Phân loại (Classification Head)**: Đi qua các lớp `GlobalAveragePooling2D` ➔ `Dropout (0.2)` ➔ `Dense (1 Unit, Sigmoid)`.
4. **Đầu ra (Output gốc)**: Trả về xác suất phân loại (0: Mèo, 1: Chó) dưới định dạng Float32.
5. **Lượng tử hóa (Quantization)**: Toàn bộ mạng được ép kiểu trọng số và activation sang số nguyên `INT8` (định dạng TFLite) giúp giảm nhẹ dung lượng và tăng tốc suy luận.

---

## 📦 Quản lý file nặng trên Git

Thư mục dữ liệu `data/` và các file mô hình huấn luyện (`*.keras`, `*.tflite`) có dung lượng lớn và đã được cấu hình trong [.gitignore](.gitignore) nhằm tránh làm nặng kho lưu trữ Git. Người sử dụng có thể tự tải dữ liệu theo link phía trên và chạy huấn luyện lại một cách nhanh chóng.
