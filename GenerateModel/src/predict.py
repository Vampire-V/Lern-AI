import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from src.preprocessing import extract_features
from pathlib import Path
import matplotlib.font_manager as fm

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_sound(file_name, model):
    features = extract_features(file_name)
    if features is not None:
        features = np.expand_dims(features, axis=0)  # เพิ่มมิติให้ข้อมูลสำหรับการทำนาย
        prediction = model.predict(features)
        return prediction[0][0]  # ส่งคืนค่าความน่าจะเป็น
    else:
        print(f"Failed to extract features for {file_name}")
        return None
    
# ฟังก์ชันสำหรับดึงรายชื่อไฟล์เสียงจากโฟลเดอร์
def load_audio_files(folder_paths):
    audio_files = []
    for folder in folder_paths:
        path = Path(folder)
        audio_files += list(path.glob('*.wav'))  # ดึงไฟล์ .wav ทั้งหมด
    return audio_files

# Example usage
if __name__ == "__main__":
    # model = load_model('models/sawasdee_model.h5')
    # prediction = predict_sound('data/new_audio.wav', model)
    # if prediction is not None:
    #     print(f"Prediction: {prediction}")

    # สร้างข้อมูลตัวอย่างสำหรับทำนาย
    # โฟลเดอร์ที่มีไฟล์เสียง
    folders = ['data']
    # โหลดรายชื่อไฟล์เสียง
    audio_files = load_audio_files(folders)
    # audio_files = ['data/sawasdee/file1.wav', 'data/sawasdee/file2.wav', 'data/other_sounds/file3.wav']

    # โหลดโมเดล
    model = load_model('models/sawasdee_model.h5')

    # ทำนายผลและเก็บผลลัพธ์
    predictions = []
    file_names = []
    print(audio_files)
    for file in audio_files:
        pred = predict_sound(str(file), model)  # แปลง Path object เป็น string ก่อนส่งให้ predict_sound
        if pred is not None:
            predictions.append(pred)
            file_names.append(file.name)  # เก็บแค่ชื่อไฟล์ไม่เอา path เต็ม

    # ตั้งค่าฟอนต์ภาษาไทย
    font_path = 'THSarabunNew.ttf'  # ระบุ path ของฟอนต์ภาษาไทย
    font_prop = fm.FontProperties(fname=font_path)
    # สร้างกราฟเพื่อตรวจสอบผลลัพธ์
    plt.figure(figsize=(10, 6))
    plt.bar(file_names, predictions, color='skyblue')
    plt.xlabel('ไฟล์เสียง', fontproperties=font_prop)
    plt.ylabel('ความน่าจะเป็น', fontproperties=font_prop)
    plt.title('ผลการทำนายไฟล์เสียงเป็น "สวัสดี"', fontproperties=font_prop)
    plt.ylim(0, 1)  # ความน่าจะเป็นอยู่ในช่วง 0 ถึง 1
    plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
    plt.tight_layout()
    plt.show()