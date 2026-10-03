# 🐾 Hướng Dẫn Tái Hiện: Phân Loại Chó & Mèo với ResNet50 (Transfer Learning & INT8 Quantization)

Dự án này cung cấp quy trình hoàn chỉnh từ **huấn luyện mô hình Transfer Learning (ResNet50)** phân loại Chó/Mèo, sau đó thực hiện **Lượng tử hóa nguyên INT8 (Post-Training Quantization - PTQ)** sang định dạng TensorFlow Lite để tối ưu triển khai trên các thiết bị nhúng và AIoT.

---

## 📋 Mục lục
1. [Cấu trúc dự án](#-cấu-trúc-dự-án)
2. [Hướng dẫn Clone Git](#-hướng-dẫn-clone-git)
3. [Cài đặt môi trường](#-cài-đặt-môi-trường)
4. [Tải và chuẩn bị Dataset](#-tải-và-chuẩn-bị-dataset)
5. [Các bước thực thi chi tiết](#-các-bước-thực-thi-chi-tiết-chạy-với-docker-hoặc-local)
6. [Kết quả Benchmark thực tế](#-kết-quả-benchmark-thực-tế)
7. [Kiến trúc mô hình](#-kiến-trúc-mô-hình)

---

## 📁 Cấu trúc dự án

```
AIOT/
├── docker-compose.yml            # File cấu hình luồng chạy Docker tự động
├── Dockerfile                    # File cấu hình môi trường Docker
├── requirements.txt              # Danh sách các thư viện Python
├── download_data.py              # Script tự động tải dataset
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

## 📥 Hướng dẫn Clone Git

Trước khi cài đặt và chạy thử nghiệm, bạn cần tải dự án về máy cục bộ. Hãy mở terminal và chạy:

```bash
git clone https://github.com/imnosuke/aiot_project.git  # Thay bằng URL Git thực tế của bạn
cd AIOT
```

---

## ⚙️ Cài đặt môi trường

Nếu bạn chạy trực tiếp trên máy (Local) thay vì dùng Docker, dự án đã chuẩn bị sẵn file `requirements.txt`. Cài đặt nhanh qua lệnh sau:

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

## 🚀 Các bước thực thi chi tiết

Dự án hỗ trợ 2 cách tiếp cận để thực thi: chạy hoàn toàn tự động bằng Docker hoặc tự chạy thủ công từng bước. 

### 🐳 Cách 1: Tự động hoàn toàn bằng Docker Compose
Dự án đã cấu hình sẵn file `docker-compose.yml` để tự động hóa toàn bộ quá trình. 

Đầu tiên, bạn cần build (hoặc rebuild) image của dự án:
```bash
docker compose build
```

Sau đó, để chạy xuyên suốt toàn bộ luồng (Tải dữ liệu ➔ Huấn luyện ➔ Lượng tử hóa & Benchmark), hãy dùng lệnh:
```bash
docker compose up
```
> *Mẹo: Bạn có thể gộp chung 2 thao tác trên bằng lệnh ngắn gọn `docker compose up --build`*

---

### 💻 Cách 2: Chạy thủ công từng bước (Download ➔ Train ➔ Quantize)
Nếu bạn muốn tự mình kiểm soát từng công đoạn, hãy lần lượt thực thi 3 bước dưới đây. (Lưu ý: Các lệnh ví dụ đang ở dạng chạy bằng Python cục bộ. Nếu bạn dùng Docker, có thể thay bằng `docker compose run --no-deps <tên-service>`).

**Bước 1: Tải và chuẩn bị Dataset (Download)**
Tập dữ liệu gốc sẽ được tự động lấy từ 🔗 [Link Github Dataset](https://github.com/guilhermedom/resnet50-transfer-learning-cats-and-dogs/tree/main/data/raw).
Dữ liệu được tải về và tự động giải nén vào thư mục `data/` với cấu trúc ảnh phân chia rõ ràng.
```bash
python download_data.py
```

**Bước 2: Huấn luyện mô hình (Train)**
Sử dụng phương pháp Transfer Learning với ResNet50 (đóng băng trọng số gốc, thêm phần đầu phân loại mới `GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(1)`).
Mô hình sau khi huấn luyện sẽ được xuất ra file `dog_cat_resnet50.keras`.
```bash
python train_dog_cat.py
```

**Bước 3: Lượng tử hóa INT8 & Đo hiệu năng (Quantize & Benchmark)**
Thực hiện quá trình Lượng tử hóa nguyên (Post-Training Integer Quantization - PTQ) dựa trên tập dữ liệu mồi, xuất ra file `resnet_cat_dog_quantized.tflite`.
Script sau đó sẽ chạy benchmark so sánh tốc độ, dung lượng và tiêu thụ RAM trực tiếp trên tập test chuẩn `microsoft/cats_vs_dogs` của HuggingFace.
```bash
python benchmark_hf.py
```

---

## 📊 Kết quả Benchmark thực tế

Kết quả đo đạc trực tiếp từ quá trình chạy thử nghiệm nghiệm thu:

=================================================================  
**BÁO CÁO BENCHMARK: RESNET50 (CHÓ/MÈO) GỐC VS QUANTIZED**  
=================================================================

| Tiêu chí | Mô hình Keras (.keras) | Mô hình TFLite Quantized | So sánh & Đánh giá |
| :--- | :---: | :---: | :--- |
| **Dung lượng file (Model Size)** | **90.63 MB** | **23.13 MB** | 🔻 **Giảm ~74.5% dung lượng** |
| **Độ chính xác (Accuracy)** | **96.00%** | **95.50%** | 🎯 **Chỉ chênh lệch 0.50%** |
| **Thời gian phản hồi/ảnh (Latency)** | **192.53 ms** | **23.29 ms** | ⚡ **Tăng tốc phản hồi ~8.2x (nhanh hơn rất nhiều)** |
| **Throughput (FPS)** | **~5.19 frames/s** | **~42.93 frames/s** | 🚀 **Xử lý lượng ảnh lớn hơn gấp ~8.2 lần** |
| **RAM tăng thêm (load+infer)** | **~193.82 MB** | **~188.13 MB** | ⚖️ **Tương đương nhau lúc chạy** |
| **RAM đỉnh process (peak)** | **~1092.63 MB** | **~1074.50 MB** | 💾 **TFLite tối ưu bộ nhớ tổng tốt hơn** |
| **Độ phức tạp (OPs)** | **7.751 GFLOPs** | **7.751 GOPs (INT8)** | ⚙️ **Chuyển Floating-point sang Integer** |

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
