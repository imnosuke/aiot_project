import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import numpy as np
import time
import os
import psutil
from datasets import load_dataset

NUM_TEST_IMAGES = 200
NUM_CALIB_IMAGES = 150 # Số ảnh mồi lấy từ tập Train

# ==============================================================================
# 1. CHUẨN BỊ TẬP DỮ LIỆU ĐẠI DIỆN (REPRESENTATIVE DATASET) TỪ TẬP TRAIN LOCAL
# ==============================================================================
print("Đang chuẩn bị tập dữ liệu mồi (Calibration) từ thư mục local...")
base_dir = r"C:\Users\PC\Desktop\Nosuke\AIOT\data"
cats_train_dir = os.path.join(base_dir, "cats_training", "cats")
dogs_train_dir = os.path.join(base_dir, "dogs_training", "dogs")

def process_local_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    return preprocess_input(img)

def create_representative_dataset():
    cat_files = tf.data.Dataset.list_files(os.path.join(cats_train_dir, "*.jpg"))
    dog_files = tf.data.Dataset.list_files(os.path.join(dogs_train_dir, "*.jpg"))
    # Gộp ảnh chó và mèo, xáo trộn để đảm bảo cân bằng
    ds = cat_files.concatenate(dog_files).shuffle(buffer_size=1000)
    ds = ds.map(process_local_path, num_parallel_calls=tf.data.AUTOTUNE)
    return ds

calib_ds = create_representative_dataset()

# ==============================================================================
# 2. CHUẨN BỊ TẬP DỮ LIỆU ĐÁNH GIÁ (BENCHMARK DATASET) TỪ HUGGINGFACE
# ==============================================================================
print("Đang tải dataset 'microsoft/cats_vs_dogs' từ HuggingFace để test...")
ds_hf = load_dataset("microsoft/cats_vs_dogs", split="train")

def preprocess_hf_example(example):
    pil_img = example['image']
    label = example['labels'] # 0: Cat, 1: Dog
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    img_arr = np.array(pil_img)
    img_resized = tf.image.resize(img_arr, [224, 224])
    return preprocess_input(img_resized), label

processed_images = []
processed_labels = []
for i in range(NUM_TEST_IMAGES):
    img, lbl = preprocess_hf_example(ds_hf[i])
    processed_images.append(img)
    processed_labels.append(lbl)

benchmark_ds = tf.data.Dataset.from_tensor_slices((processed_images, processed_labels))

# ==============================================================================
# 3. LOAD MÔ HÌNH KERAS ĐÃ HUẤN LUYỆN
# ==============================================================================
keras_model_path = r"C:\Users\PC\Desktop\Nosuke\AIOT\dog_cat_resnet50.keras"
print(f"\n[1/3] Đang tải mô hình ResNet50 phân loại Chó/Mèo: {keras_model_path}...")
fine_tuned_model = tf.keras.models.load_model(keras_model_path)


# ==============================================================================
# 4. HÀM BENCHMARK MÔ HÌNH KERAS
# ==============================================================================
def benchmark_keras(model, dataset, model_path):
    print("Bắt đầu đo Keras (Original)...")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    total_time = 0
    correct = 0
    for image, label in dataset:
        img_batch = tf.expand_dims(image, 0)
        
        start_time = time.time()
        pred = model.predict(img_batch, verbose=0)
        end_time = time.time()
        
        total_time += (end_time - start_time)
        pred_label = 1 if pred[0][0] > 0.5 else 0
        if pred_label == label.numpy():
            correct += 1
            
    mem_after = process.memory_info().rss / (1024 * 1024)
    ram_mb = max(0, mem_after - mem_before)
    
    avg_latency = (total_time / NUM_TEST_IMAGES) * 1000
    throughput = NUM_TEST_IMAGES / total_time
    accuracy = correct / NUM_TEST_IMAGES
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    return accuracy, avg_latency, throughput, size_mb, ram_mb


# ==============================================================================
# 5. QUANTIZATION SANG TFLITE
# ==============================================================================
def quantize_to_tflite(model, output_path):
    print(f"\n[2/3] Bắt đầu Quantization INT8 (Dùng {NUM_CALIB_IMAGES} ảnh từ tập Train để mồi)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Dùng hàm này để nạp dữ liệu từ thư mục local training
    def representative_data_gen():
        for image in calib_ds.take(NUM_CALIB_IMAGES):
            yield [tf.expand_dims(image, 0)]
            
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Đã lưu: {output_path}")
    return output_path


# ==============================================================================
# 6. HÀM BENCHMARK MÔ HÌNH TFLITE
# ==============================================================================
def benchmark_tflite(tflite_path, dataset):
    print(f"\n[3/3] Bắt đầu đo TFLite (Quantized) trên tập HuggingFace...")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    total_time = 0
    correct = 0
    
    for image, label in dataset:
        img_batch = tf.expand_dims(image, 0)
        interpreter.set_tensor(input_index, img_batch)
        
        start_time = time.time()
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)
        end_time = time.time()
        
        total_time += (end_time - start_time)
        pred_label = 1 if output_data[0][0] > 0.5 else 0
        if pred_label == label.numpy():
            correct += 1
            
    mem_after = process.memory_info().rss / (1024 * 1024)
    ram_mb = max(0, mem_after - mem_before)
    
    avg_latency = (total_time / NUM_TEST_IMAGES) * 1000
    throughput = NUM_TEST_IMAGES / total_time
    accuracy = correct / NUM_TEST_IMAGES
    size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    return accuracy, avg_latency, throughput, size_mb, ram_mb


def calculate_flops(model):
    """Tính toán tổng số FLOPs của một mô hình Keras"""
    # Khai báo TensorSpec với kích thước đầu vào của ResNet50 (Batch=1, 224, 224, 3)
    input_signature = [tf.TensorSpec([1, 224, 224, 3], tf.float32)]
    
    # Chuyển đổi mô hình thành concrete function
    forward_graph = tf.function(model).get_concrete_function(input_signature)
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(forward_graph)
    
    # Tính toán FLOPs bằng Profiler
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
        # Mute logging output bằng cách thêm thuộc tính ẩn
        opts['output'] = 'none' 
        
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
        
        return flops.total_float_ops

# ==============================================================================
# 7. THỰC THI VÀ IN KẾT QUẢ
# ==============================================================================
tflite_model_path = r"C:\Users\PC\Desktop\Nosuke\AIOT\resnet_cat_dog_quantized.tflite"

k_acc, k_lat, k_throughput, k_size, k_ram = benchmark_keras(fine_tuned_model, benchmark_ds, keras_model_path)
quantize_to_tflite(fine_tuned_model, tflite_model_path)
t_acc, t_lat, t_throughput, t_size, t_ram = benchmark_tflite(tflite_model_path, benchmark_ds)

print("\n" + "="*65)
print(" BÁO CÁO BENCHMARK: RESNET50 (CHÓ/MÈO) GỐC VS QUANTIZED")
print("="*65)
print(f"{'Tiêu chí':<25} | {'Mô hình Keras (.keras)':<22} | {'Mô hình TFLite Quantized':<22}")
print("-"*65)
print(f"{'Dung lượng File':<25} | {k_size:.2f} MB{'':<15} | {t_size:.2f} MB")
print(f"{'Độ chính xác (Accuracy)':<25} | {k_acc*100:.2f}%{'':<16} | {t_acc*100:.2f}%")
print(f"{'Thời gian phản hồi/ảnh':<25} | {k_lat:.2f} ms{'':<15} | {t_lat:.2f} ms")
print(f"{'Throughput (FPS)':<25} | {k_throughput:.2f} frames/s{'':<7} | {t_throughput:.2f} frames/s")
print(f"{'Tiêu thụ RAM (Tăng thêm)':<25} | ~{k_ram:.2f} MB{'':<14} | ~{t_ram:.2f} MB")
k_flops_keras = calculate_flops(fine_tuned_model)
# k_flops_tflite = calculate_flops(tf.lite.Interpreter(model_path=tflite_model_path))
print(f"{'Tổng FLOPs':<25} | {k_flops_keras / 1e9:.3f} GFLOPs{'':<13} | (Mô hình keras)")
print(f"{'Độ phức tạp (OPs)':<25} | {k_flops_keras / 1e9:.3f} GOPs (INT8){'':<7} | (Mô hình TFLite Quantized)")


print("="*65)
print("* Căn chỉnh Quantization bằng 150 ảnh từ thư mục training local.")
print("* Đánh giá Benchmark bằng 200 ảnh từ HuggingFace 'microsoft/cats_vs_dogs'.")
