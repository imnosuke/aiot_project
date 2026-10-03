"""
Benchmark ResNet50 (Chó/Mèo): Keras gốc vs TFLite INT8.

Kiến trúc: script chạy ở chế độ "orchestrator" (mặc định) và gọi lại chính nó
bằng các process con riêng biệt (--mode keras | quantize | tflite). Nhờ vậy RAM
của mỗi mô hình được đo trong một process sạch, không bị nhiễu bởi mô hình kia.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

DATA_DIR = os.environ.get("DATA_DIR", "data")
MODEL_DIR = os.environ.get("MODEL_DIR", ".")
NUM_TEST_IMAGES = 500
NUM_CALIB_IMAGES = 150  # Số ảnh mồi lấy từ tập Train

KERAS_PATH = os.path.join(MODEL_DIR, "dog_cat_resnet50.keras")
TFLITE_PATH = os.path.join(MODEL_DIR, "resnet_cat_dog_quantized.tflite")


# ==============================================================================
# TIỆN ÍCH ĐO RAM
# ==============================================================================
def rss_mb():
    """RSS hiện tại của process (MB)."""
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def peak_rss_mb():
    """RSS đỉnh của process (MB)."""
    try:
        import resource  # Linux / Docker
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:  # Windows
        import psutil
        info = psutil.Process(os.getpid()).memory_info()
        return getattr(info, "peak_wset", info.rss) / (1024 * 1024)


# ==============================================================================
# DỮ LIỆU
# ==============================================================================
def build_calib_ds():
    """Tập dữ liệu mồi (Calibration) từ thư mục train local."""
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input

    cats_train_dir = os.path.join(DATA_DIR, "cats_training", "cats")
    dogs_train_dir = os.path.join(DATA_DIR, "dogs_training", "dogs")

    def process_local_path(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        return preprocess_input(img)

    cat_files = tf.data.Dataset.list_files(os.path.join(cats_train_dir, "*.jpg"))
    dog_files = tf.data.Dataset.list_files(os.path.join(dogs_train_dir, "*.jpg"))
    ds = cat_files.concatenate(dog_files).shuffle(buffer_size=1000)
    return ds.map(process_local_path, num_parallel_calls=tf.data.AUTOTUNE)


def build_benchmark_ds():
    """Tập đánh giá từ HuggingFace 'microsoft/cats_vs_dogs'."""
    import numpy as np
    import tensorflow as tf
    from datasets import load_dataset
    from tensorflow.keras.applications.resnet50 import preprocess_input

    print("Đang tải dataset 'microsoft/cats_vs_dogs' từ HuggingFace để test...")
    ds_hf = load_dataset("microsoft/cats_vs_dogs", split="train")

    images, labels = [], []
    for i in range(NUM_TEST_IMAGES):
        ex = ds_hf[i]
        pil_img = ex["image"]
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img = tf.image.resize(np.array(pil_img), [224, 224])
        images.append(preprocess_input(img))
        labels.append(ex["labels"])  # 0: Cat, 1: Dog
    return tf.data.Dataset.from_tensor_slices((images, labels))


# ==============================================================================
# CÁC CHẾ ĐỘ CHẠY TRONG PROCESS CON
# ==============================================================================
def calculate_flops(model):
    """Tính tổng số FLOPs của một mô hình Keras."""
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    input_signature = [tf.TensorSpec([1, 224, 224, 3], tf.float32)]
    forward_graph = tf.function(model).get_concrete_function(input_signature)
    _, graph_def = convert_variables_to_constants_v2_as_graph(forward_graph)

    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts["output"] = "none"
        flops = tf.compat.v1.profiler.profile(
            graph=graph, run_meta=run_meta, cmd="op", options=opts
        )
        return flops.total_float_ops


def run_keras():
    import tensorflow as tf

    dataset = build_benchmark_ds()

    # Baseline: sau khi import + chuẩn bị dữ liệu, TRƯỚC khi load mô hình
    mem_before = rss_mb()

    print(f"\n[1/3] Đang tải mô hình Keras: {KERAS_PATH}...")
    model = tf.keras.models.load_model(KERAS_PATH)

    total_time, correct = 0.0, 0
    for image, label in dataset:
        batch = tf.expand_dims(image, 0)
        start = time.time()
        pred = model.predict(batch, verbose=0)
        total_time += time.time() - start
        if (1 if pred[0][0] > 0.5 else 0) == label.numpy():
            correct += 1

    mem_after = rss_mb()
    peak = peak_rss_mb()

    # FLOPs tính sau khi đo RAM để không làm nhiễu số liệu
    flops = calculate_flops(model)

    return {
        "accuracy": correct / NUM_TEST_IMAGES,
        "latency_ms": total_time / NUM_TEST_IMAGES * 1000,
        "throughput": NUM_TEST_IMAGES / total_time,
        "size_mb": os.path.getsize(KERAS_PATH) / (1024 * 1024),
        "ram_mb": max(0.0, mem_after - mem_before),
        "peak_mb": peak,
        "flops": flops,
    }


def run_quantize():
    import tensorflow as tf

    calib_ds = build_calib_ds()
    print(f"\n[2/3] Bắt đầu Quantization INT8 (Dùng {NUM_CALIB_IMAGES} ảnh từ tập Train để mồi)...")
    model = tf.keras.models.load_model(KERAS_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for image in calib_ds.take(NUM_CALIB_IMAGES):
            yield [tf.expand_dims(image, 0)]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"Đã lưu: {TFLITE_PATH}")
    return {}


def run_tflite():
    import tensorflow as tf

    dataset = build_benchmark_ds()

    # Baseline: sau khi import + chuẩn bị dữ liệu, TRƯỚC khi tạo Interpreter
    mem_before = rss_mb()

    print("\n[3/3] Bắt đầu đo TFLite (Quantized) trên tập HuggingFace...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    
    # In ra một số Scale và Zero Point để kiểm tra
    print("\n--- THÔNG TIN LƯỢNG TỬ HÓA (SCALE & ZERO POINT) ---")
    count = 0
    for tensor in interpreter.get_tensor_details():
        scale, zero_point = tensor['quantization']
        if scale > 0.0:  # Chỉ in những tensor đã được quantize sang INT8
            print(f"Lớp: {tensor['name'][:40]:<40} | Scale: {scale:.6f} | Zero Point: {zero_point}")
            count += 1
            if count >= 5:  # In 5 tensor đại diện để tránh trôi màn hình
                break
    print("---------------------------------------------------\n")

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    total_time, correct = 0.0, 0
    for image, label in dataset:
        batch = tf.expand_dims(image, 0)
        interpreter.set_tensor(input_index, batch)
        start = time.time()
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        total_time += time.time() - start
        if (1 if output[0][0] > 0.5 else 0) == label.numpy():
            correct += 1

    mem_after = rss_mb()

    return {
        "accuracy": correct / NUM_TEST_IMAGES,
        "latency_ms": total_time / NUM_TEST_IMAGES * 1000,
        "throughput": NUM_TEST_IMAGES / total_time,
        "size_mb": os.path.getsize(TFLITE_PATH) / (1024 * 1024),
        "ram_mb": max(0.0, mem_after - mem_before),
        "peak_mb": peak_rss_mb(),
    }


MODES = {"keras": run_keras, "quantize": run_quantize, "tflite": run_tflite}


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================
def run_in_subprocess(mode):
    """Chạy một chế độ trong process riêng và đọc kết quả qua file JSON tạm."""
    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        cmd = [sys.executable, os.path.abspath(__file__), "--mode", mode, "--out", out_path]
        subprocess.run(cmd, check=True)
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def print_report(k, t):
    print("\n" + "=" * 65)
    print(" BÁO CÁO BENCHMARK: RESNET50 (CHÓ/MÈO) GỐC VS QUANTIZED")
    print("=" * 65)
    print(f"{'Tiêu chí':<25} | {'Mô hình Keras (.keras)':<22} | {'Mô hình TFLite Quantized':<22}")
    print("-" * 65)
    print(f"{'Dung lượng File':<25} | {k['size_mb']:.2f} MB{'':<15} | {t['size_mb']:.2f} MB")
    print(f"{'Độ chính xác (Accuracy)':<25} | {k['accuracy']*100:.2f}%{'':<16} | {t['accuracy']*100:.2f}%")
    print(f"{'Thời gian phản hồi/ảnh':<25} | {k['latency_ms']:.2f} ms{'':<15} | {t['latency_ms']:.2f} ms")
    print(f"{'Throughput (FPS)':<25} | {k['throughput']:.2f} frames/s{'':<7} | {t['throughput']:.2f} frames/s")
    print(f"{'RAM tăng thêm (load+infer)':<25} | ~{k['ram_mb']:.2f} MB{'':<14} | ~{t['ram_mb']:.2f} MB")
    print(f"{'RAM đỉnh process (peak)':<25} | ~{k['peak_mb']:.2f} MB{'':<14} | ~{t['peak_mb']:.2f} MB")
    print(f"{'Tổng FLOPs':<25} | {k['flops'] / 1e9:.3f} GFLOPs{'':<13} | (Mô hình keras)")
    print(f"{'Độ phức tạp (OPs)':<25} | {k['flops'] / 1e9:.3f} GOPs (INT8){'':<7} | (Mô hình TFLite Quantized)")
    print("=" * 65)
    print("* Mỗi mô hình được đo trong một process riêng, baseline lấy trước khi load mô hình.")
    print(f"* Căn chỉnh Quantization bằng {NUM_CALIB_IMAGES} ảnh từ thư mục training local.")
    print(f"* Đánh giá Benchmark bằng {NUM_TEST_IMAGES} ảnh từ HuggingFace 'microsoft/cats_vs_dogs'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES), help="Chạy một chế độ (dùng nội bộ)")
    parser.add_argument("--out", help="Đường dẫn file JSON để ghi kết quả")
    args = parser.parse_args()

    if args.mode:  # Process con
        result = MODES[args.mode]()
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f)
        return

    # Orchestrator: mỗi bước một process riêng
    keras_res = run_in_subprocess("keras")
    run_in_subprocess("quantize")
    tflite_res = run_in_subprocess("tflite")
    print_report(keras_res, tflite_res)


if __name__ == "__main__":
    main()
