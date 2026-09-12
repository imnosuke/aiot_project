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

---

## 🗂️ Tải và chuẩn bị Dataset

Tập dữ liệu ảnh dùng cho huấn luyện và kiểm thử được lấy từ repository:
🔗 **Link Dataset:** [guilhermedom/resnet50-transfer-learning-cats-and-dogs (data/raw)](https://github.com/guilhermedom/resnet50-transfer-learning-cats-and-dogs/tree/main/data/raw)

### Hướng dẫn thiết lập:
1. Tải các thư mục ảnh tương ứng trong link trên về máy.
2. Tạo thư mục `data/` trong thư mục gốc của dự án `AIOT/` và sắp xếp các file ảnh theo đúng cấu trúc sau:
   ```
   AIOT/
   └── data/
       ├── cats_training/cats/*.jpg
       ├── dogs_training/dogs/*.jpg
       ├── cats_testing/cats/*.jpg
       └── dogs_testing/dogs/*.jpg
   ```

---

## 🚀 Các bước thực thi chi tiết

### Bước 1: Huấn luyện mô hình (Fine-Tuning ResNet50)
Chạy script huấn luyện:
```bash
python train_dog_cat.py
```
* **Cơ chế hoạt động:**
  - Tải kiến trúc backbone **ResNet50** đã pretrained trên tập ImageNet (`include_top=False`).
  - Đóng băng (freeze) các tầng gốc để giữ các đặc trưng bậc thấp.
  - Gắn classification head mới: `GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(1, activation='sigmoid')`.
  - Tự động kích hoạt GPU (nếu máy có GPU NVIDIA hỗ trợ CUDA) với chế độ cấp phát động `memory_growth`.
  - Huấn luyện qua 5 epochs và tự động lưu mô hình tại: `dog_cat_resnet50.keras`.

---

### Bước 2: Lượng tử hóa INT8 & Benchmark hiệu năng
Chạy script lượng tử hóa và đánh giá:
```bash
python benchmark_hf.py
```
* **Cơ chế hoạt động:**
  1. **Calibration Dataset (Dữ liệu mồi):** Lấy ngẫu nhiên 150 ảnh từ thư mục training local để mô phỏng dải phân bố động của activation tensor.
  2. **Benchmark Dataset:** Tải 200 ảnh chuẩn từ dataset `microsoft/cats_vs_dogs` trên HuggingFace để kiểm tra độ chính xác độc lập.
  3. **Đo đạc mô hình gốc:** Tính toán độ trễ trung bình (latency), dung lượng file, và accuracy của file `.keras`.
  4. **Post-Training Quantization (PTQ):** Ép toàn bộ trọng số và activation sang kiểu số nguyên 8-bit (`INT8`), xuất ra file `resnet_cat_dog_quantized.tflite`.
  5. **Đo đạc mô hình Quantized:** Đánh giá độ trễ và độ chính xác của mô hình TFLite INT8 trên cùng tập test.

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
| **Thời gian phản hồi/ảnh (Latency)** | **98.70 ms** | **89.62 ms** | ⚡ **Tăng tốc phản hồi ~9.2%** |

> 📌 **Nhận xét chuyên môn:**
> - Mô hình sau khi lượng tử hóa sang INT8 tiết kiệm gần **75% bộ nhớ lưu trữ**, cực kỳ lý tưởng để nạp vào ROM/Flash của các board mạch nhúng, thiết bị Edge AI (Raspberry Pi, Jetson Nano, Coral Edge TPU).
> - Độ chính xác gần như được bảo toàn trọn vẹn (96.00% xuống 95.50%), chứng minh dữ liệu calibration đại diện rất tốt cho miền bài toán.
> - Trên phần cứng chuyên dụng có tập lệnh xử lý phép tính số nguyên (INT8 SIMD / Tensor Cores / NPU), tốc độ xử lý sẽ tăng tốc vượt bậc so với vi xử lý thông thường.

---

## 🧱 Kiến trúc mô hình

```
Input Tensor (224, 224, 3)
      │
      ▼
ResNet50 Backbone (Pretrained ImageNet, Frozen)
      │
      ▼
GlobalAveragePooling2D
      │
      ▼
Dropout (Rate = 0.2)
      │
      ▼
Dense (1 Unit, Activation = Sigmoid)
      │
      ▼
Output: Xác suất [0, 1] (0: Mèo / Cat, 1: Chó / Dog)
```

---

## 📦 Quản lý file nặng trên Git

Thư mục dữ liệu `data/` và các file mô hình huấn luyện (`*.keras`, `*.tflite`) có dung lượng lớn và đã được cấu hình trong [.gitignore](.gitignore) nhằm tránh làm nặng kho lưu trữ Git. Người sử dụng có thể tự tải dữ liệu theo link phía trên và chạy huấn luyện lại một cách nhanh chóng.
