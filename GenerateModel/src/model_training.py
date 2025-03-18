import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.preprocessing import extract_features

def load_data_from_folder(data_folder):
    """
    ฟังก์ชันสำหรับโหลดไฟล์เสียงทั้งหมดจากโฟลเดอร์ที่กำหนด
    และแปลงเป็นคุณลักษณะเสียง (features)
    """
    file_paths = []
    labels = []

    data_folder = Path(data_folder)  # ใช้ Path ในการจัดการ path
    for category in ['sawasdee', 'other_sounds']:
        folder = data_folder / category  # ใช้ '/' ในการเชื่อม path
        label = 1 if category == 'sawasdee' else 0

        for file in folder.glob('*.wav'):  # ใช้ glob เพื่อหาไฟล์ .wav
            file_paths.append(str(file))  # แปลง path เป็น string
            labels.append(label)
    print(file_paths)
    print(f"Loaded {len(file_paths)} files-------------------------------------------------------")
    return file_paths, labels

def prepare_features_and_labels(file_paths, labels):
    """
    แปลงไฟล์เสียงจาก file_paths เป็น features (MFCCs) และจัดเตรียม labels
    """
    features = []

    for file_name in file_paths:
        data = extract_features(file_name)
        if data is not None:
            print(f"Feature shape for {file_name}: {data.shape}")
            features.append(data)
        else:
            print(f"Failed to extract features for {file_name}")
    
    # แปลงเป็น NumPy array
    X = np.array(features)
    y = np.array(labels)

    return X, y

def train_model(X_train, y_train):
    """
    ฟังก์ชันสำหรับสร้างและฝึกโมเดล
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    return model

def save_model(model, path):
    """
    บันทึกโมเดลที่ฝึกแล้ว
    """
    model.save(path)

# จุดเริ่มต้นของโปรแกรม (Main script)
if __name__ == "__main__":
    # ระบุโฟลเดอร์หลักที่มีไฟล์เสียงสำหรับการฝึก
    data_folder = 'data/'  # โฟลเดอร์หลักที่มีโฟลเดอร์ย่อย sawasdee และ other_sounds

    # โหลดข้อมูลจากโฟลเดอร์
    file_paths, labels = load_data_from_folder(data_folder)

    # เตรียม features และ labels
    X, y = prepare_features_and_labels(file_paths, labels)

    # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ฝึกโมเดล
    model = train_model(X_train, y_train)

    # บันทึกโมเดล
    save_model(model, 'models/sawasdee_model.h5')

    # ประเมินผลลัพธ์
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")
