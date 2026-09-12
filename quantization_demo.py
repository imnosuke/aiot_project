import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Tắt thông báo log rác từ TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_or_load_model():
    """
    Tải mô hình ResNet50 pre-trained.
    Kích thước ảnh đầu vào chuẩn là 224x224x3.
    """
    print("\n[1] Đang tải mô hình Keras ResNet50 (Pre-trained ImageNet)...")
    model = tf.keras.applications.ResNet50(
        include_top=True,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    return model

# ---------------------------------------------------------
# PHẦN CHUẨN BỊ DỮ LIỆU CÔNG BẰNG (FAIR DATASET: IMAGENET-V2)
# ---------------------------------------------------------
print("\n[*] Đang kết nối tải tập dữ liệu ImageNet-V2 (Chỉ tải ~1.2GB ở lần chạy đầu tiên)...")
# Lấy 100 ảnh đầu tiên làm Representative Dataset (Hiệu chuẩn)
dataset_calib = tfds.load('imagenet_v2', split='test[:1000]', as_supervised=True)
# Lấy 100 ảnh tiếp theo (từ 100 đến 200) làm tập Test (hoàn toàn tách biệt)
dataset_test = tfds.load('imagenet_v2', split='test[1000:2000]', as_supervised=True)

def representative_dataset_gen():
    """
    Tập dữ liệu mẫu đại diện (Representative Dataset) sử dụng ảnh THẬT từ ImageNet-V2.
    """
    for img, label in dataset_calib:
        # Resize ảnh về 224x224
        img = tf.image.resize(img, (224, 224))
        # Tiền xử lý chuẩn của ResNet50 (chuyển RGB->BGR, Zero-center)
        img = tf.keras.applications.resnet50.preprocess_input(img.numpy())
        # Thêm batch dimension (1, 224, 224, 3)
        img = np.expand_dims(img, axis=0)
        yield [img]

def convert_to_fp32_tflite(model, output_path="resnet50_fp32.tflite"):
    print(f"\n[2] Đang chuyển đổi mô hình sang TFLite FP32 -> {output_path}...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_fp32_model)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    -> Tạo thành công {output_path} ({size_mb:.2f} MB)")
    return output_path

def convert_to_int8_tflite(model, output_path="resnet50_int8.tflite"):
    print(f"\n[3] Đang chuyển đổi mô hình sang TFLite INT8 (Integer-only) -> {output_path}...")
    print("    (Đang chạy Calibration trên 100 ảnh thực tế ImageNet-V2, vui lòng đợi...)")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_int8_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_int8_model)
        
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    -> Tạo thành công {output_path} ({size_mb:.2f} MB)")
    return output_path

def benchmark_models(fp32_path="resnet50_fp32.tflite", int8_path="resnet50_int8.tflite"):
    print("\n" + "="*70)
    print("      BẮT ĐẦU KIỂM THỬ VÀ SO SÁNH HIỆU NĂNG TRÊN IMAGENET-V2")
    print("="*70)
    
    # 1. SO SÁNH DUNG LƯỢNG FILE
    size_fp32 = os.path.getsize(fp32_path)
    size_int8 = os.path.getsize(int8_path)
    reduction_pct = (1.0 - (size_int8 / size_fp32)) * 100.0
    
    print("\n--- 1. SO SÁNH DUNG LƯỢNG TỆP MÔ HÌNH ---")
    print(f"• FP32 Model Size: {size_fp32 / (1024*1024):.2f} MB")
    print(f"• INT8 Model Size: {size_int8 / (1024*1024):.2f} MB")
    print(f"• Tỷ lệ giảm dung lượng: {reduction_pct:.2f}%")

    # 2. KHỞI TẠO INTERPRETER
    interp_fp32 = tf.lite.Interpreter(model_path=fp32_path)
    interp_fp32.allocate_tensors()
    fp32_in_details = interp_fp32.get_input_details()[0]
    fp32_out_details = interp_fp32.get_output_details()[0]

    interp_int8 = tf.lite.Interpreter(model_path=int8_path)
    interp_int8.allocate_tensors()
    int8_in_details = interp_int8.get_input_details()[0]
    int8_out_details = interp_int8.get_output_details()[0]

    in_scale, in_zero_point = int8_in_details['quantization']
    out_scale, out_zero_point = int8_out_details['quantization']

    # 3. CHUẨN BỊ DỮ LIỆU TEST (100 ảnh từ ImageNet-V2)
    test_images = []
    test_labels = []
    print("\n[Đang tiền xử lý tập 100 ảnh Test (Unseen Data)...]")
    for img, label in dataset_test:
        img = tf.image.resize(img, (224, 224))
        img = tf.keras.applications.resnet50.preprocess_input(img.numpy())
        test_images.append(img)
        test_labels.append(label.numpy())
        
    test_images = np.array(test_images) # Shape: (100, 224, 224, 3)
    test_labels = np.array(test_labels) # Shape: (100,)
    num_samples = len(test_images)

    # 4. ĐO ĐỘ TRỄ SUY LUẬN (LATENCY)
    print(f"\n--- 2. ĐO ĐỘ TRỄ SUY LUẬN (LATENCY TRÊN {num_samples} MẪU) ---")
    
    # Warmup runs (Dùng 2 samples cho warmup)
    for i in range(2):
        sample_warmup = test_images[i:i+1]
        interp_fp32.set_tensor(fp32_in_details['index'], sample_warmup)
        interp_fp32.invoke()
        
        warmup_int8_in = np.round(sample_warmup / in_scale + in_zero_point)
        warmup_int8_in = np.clip(warmup_int8_in, -128, 127).astype(np.int8)
        interp_int8.set_tensor(int8_in_details['index'], warmup_int8_in)
        interp_int8.invoke()

    # Đo latency FP32
    fp32_outputs = []
    start_time = time.perf_counter()
    for i in range(num_samples):
        sample = test_images[i:i+1]
        interp_fp32.set_tensor(fp32_in_details['index'], sample)
        interp_fp32.invoke()
        fp32_outputs.append(interp_fp32.get_tensor(fp32_out_details['index']))
    total_fp32_time = (time.perf_counter() - start_time) * 1000.0
    avg_fp32_latency = total_fp32_time / num_samples

    # Đo latency INT8
    int8_outputs_dequantized = []
    start_time = time.perf_counter()
    for i in range(num_samples):
        sample_fp32 = test_images[i:i+1]
        # Tiền xử lý lượng tử hóa
        sample_int8 = np.round(sample_fp32 / in_scale + in_zero_point)
        sample_int8 = np.clip(sample_int8, -128, 127).astype(np.int8)
        
        interp_int8.set_tensor(int8_in_details['index'], sample_int8)
        interp_int8.invoke()
        raw_int8_out = interp_int8.get_tensor(int8_out_details['index'])
        
        # Giải lượng tử hóa (Dequantization)
        dequant_out = (raw_int8_out.astype(np.float32) - out_zero_point) * out_scale
        int8_outputs_dequantized.append(dequant_out)
    total_int8_time = (time.perf_counter() - start_time) * 1000.0
    avg_int8_latency = total_int8_time / num_samples

    print(f"• FP32 Latency trung bình : {avg_fp32_latency:.4f} ms/mẫu")
    print(f"• INT8 Latency trung bình : {avg_int8_latency:.4f} ms/mẫu")
    speedup = (avg_fp32_latency / avg_int8_latency) if avg_int8_latency > 0 else 1.0
    print(f"• Tăng tốc độ suy luận   : {speedup:.2f}x")

    # 5. TÍNH TOÁN ĐỘ CHÍNH XÁC (ACCURACY) ĐỐI VỚI NHÃN THỰC TẾ
    print("\n--- 3. ĐÁNH GIÁ ĐỘ CHÍNH XÁC (GROUND TRUTH ACCURACY) ---")
    fp32_arr = np.concatenate(fp32_outputs, axis=0)
    int8_arr = np.concatenate(int8_outputs_dequantized, axis=0)

    top1_fp32 = np.argmax(fp32_arr, axis=1)
    top1_int8 = np.argmax(int8_arr, axis=1)

    acc_fp32 = np.mean(top1_fp32 == test_labels) * 100.0
    acc_int8 = np.mean(top1_int8 == test_labels) * 100.0
    acc_drop = acc_fp32 - acc_int8

    print(f"• Độ chính xác FP32 (Top-1)  : {acc_fp32:.2f}%")
    print(f"• Độ chính xác INT8 (Top-1)  : {acc_int8:.2f}%")
    print(f"• Mức sụt giảm (Acc. Drop)   : {acc_drop:.2f}% (Càng gần 0 càng tốt)")
    
    # Cosine Similarity giữa 2 vector xác suất
    dot_product = np.sum(fp32_arr * int8_arr, axis=1)
    norm_fp32 = np.linalg.norm(fp32_arr, axis=1)
    norm_int8 = np.linalg.norm(int8_arr, axis=1)
    cosine_sim = np.mean(dot_product / (norm_fp32 * norm_int8 + 1e-8))
    print(f"• Độ trùng khớp Vector (Cos) : {cosine_sim:.6f} (Gần tuyệt đối 1.0)")
    print("="*70)

if __name__ == "__main__":
    model = build_or_load_model()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fp32_file = os.path.join(base_dir, "resnet50_fp32.tflite")
    int8_file = os.path.join(base_dir, "resnet50_int8.tflite")
    
    convert_to_fp32_tflite(model, fp32_file)
    convert_to_int8_tflite(model, int8_file)
    
    benchmark_models(fp32_file, int8_file)
