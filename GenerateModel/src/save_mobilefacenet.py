import os
import tensorflow as tf
from MobileFaceNet_TF.nets.MobileFaceNetCustom import MobileFaceNetCustom

# พาธไปยังโฟลเดอร์ pretrained_model
checkpoint_dir = "./src/MobileFaceNet_TF/arch/pretrained_model"
saved_model_path_keras = "./saved_model_mobilefacenet/saved_model_mobilefacenet.keras"
saved_model_path_h5 = "./saved_model_mobilefacenet/saved_model_mobilefacenet.h5"

# ตรวจสอบโฟลเดอร์ Checkpoint
if not os.path.exists(checkpoint_dir):
    raise FileNotFoundError(f"Directory not found: {checkpoint_dir}")
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
if checkpoint_path is None:
    raise FileNotFoundError(f"No checkpoint found in directory {checkpoint_dir}")
print(f"Checkpoint found: {checkpoint_path}")

# นิยามโครงสร้างของโมเดล
def create_mobilefacenet_model():
    inputs = tf.keras.Input(shape=(112, 112, 3), name="input_image")  # ขนาดอินพุต
    model = MobileFaceNetCustom(embedding_size=128)  # กำหนด embedding size
    outputs = model(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="MobileFaceNetCustom")

try:
    # สร้างโมเดล
    model = create_mobilefacenet_model()

    # โหลดน้ำหนักด้วย tf.train.Checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_path).expect_partial()

    # บันทึกโมเดลในรูปแบบ .keras
    model.save(saved_model_path_keras)
    print(f"Model successfully saved to {saved_model_path_keras}")

    # บันทึกโมเดลในรูปแบบ .h5
    model.save(saved_model_path_h5)
    print(f"Model successfully saved to {saved_model_path_h5}")
except Exception as e:
    print(f"Error saving model: {e}")
