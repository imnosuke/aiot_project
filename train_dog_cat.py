import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# --- CẤU HÌNH SỬ DỤNG GPU ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Bật tính năng cấp phát bộ nhớ động (Memory Growth) để tránh lỗi tràn RAM GPU (OOM)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n✅ Đã tìm thấy {len(gpus)} GPU: {[gpu.name for gpu in gpus]}")
        print("Sẽ sử dụng GPU để huấn luyện siêu tốc!\n")
    except RuntimeError as e:
        print(f"Lỗi cấu hình GPU: {e}")
else:
    print("\n⚠️ Không tìm thấy GPU tương thích. Sẽ sử dụng CPU (sẽ chậm hơn).\n")
# -----------------------------

# Định nghĩa hàm đọc và tiền xử lý ảnh
def process_path(file_path, label):
    # Đọc file ảnh
    img = tf.io.read_file(file_path)
    # Giải mã ảnh JPEG
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize về 224x224 (kích thước chuẩn của ResNet50)
    img = tf.image.resize(img, [224, 224])
    # Tiền xử lý theo chuẩn của ResNet50 (căn chỉnh giá trị pixel)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label

def create_dataset(cat_dir, dog_dir):
    # Lấy danh sách tất cả các file ảnh
    cat_files = tf.data.Dataset.list_files(os.path.join(cat_dir, "*.jpg"))
    dog_files = tf.data.Dataset.list_files(os.path.join(dog_dir, "*.jpg"))
    
    # Gán nhãn: Mèo (Cat) = 0, Chó (Dog) = 1
    cat_ds = cat_files.map(lambda x: (x, 0))
    dog_ds = dog_files.map(lambda x: (x, 1))
    
    # Nối 2 tập dữ liệu lại và xáo trộn (shuffle)
    ds = cat_ds.concatenate(dog_ds)
    ds = ds.shuffle(buffer_size=2000)
    
    # Áp dụng hàm tiền xử lý
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
    return ds

# --- 1. CHUẨN BỊ DỮ LIỆU ---
print("Đang chuẩn bị dữ liệu...")
train_ds = create_dataset(
    r"C:\Users\PC\Desktop\Nosuke\AIOT\data\cats_training\cats",
    r"C:\Users\PC\Desktop\Nosuke\AIOT\data\dogs_training\dogs"
)
val_ds = create_dataset(
    r"C:\Users\PC\Desktop\Nosuke\AIOT\data\cats_testing\cats",
    r"C:\Users\PC\Desktop\Nosuke\AIOT\data\dogs_testing\dogs"
)

# --- 2. XÂY DỰNG MÔ HÌNH TRANSFER LEARNING ---
# Load ResNet50, BỎ lớp phân loại 1000 nhãn (include_top=False)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Đóng băng (Freeze) các lớp của mô hình gốc để không làm mất các đặc trưng đã được học
base_model.trainable = False

# Thêm phần đầu phân loại (Classification Head) mới cho 2 nhãn
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x) # Giảm overfitting
# Lớp đầu ra 1 node dùng hàm sigmoid để phân loại nhị phân (0 hoặc 1)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile mô hình
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# --- 3. HUẤN LUYỆN MÔ HÌNH ---
print("Bắt đầu huấn luyện mô hình (Fine-tuning)...")
# Train 5 epochs (bạn có thể tăng lên nếu muốn độ chính xác cao hơn)
model.fit(train_ds, validation_data=val_ds, epochs=5)

# --- 4. LƯU MÔ HÌNH ---
save_path = r"C:\Users\PC\Desktop\Nosuke\AIOT\dog_cat_resnet50.keras"
model.save(save_path)
print(f"Hoàn thành! Đã lưu mô hình tại: {save_path}")
