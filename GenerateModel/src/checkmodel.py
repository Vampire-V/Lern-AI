# import tensorflow as tf
# from tensorflow.keras.models import load_model

# model_path = "D:/ayodia/top-panacea/Ayodia.PanaceaHS/Ayodia.PanaceaHS.WebAPI/Models/facenet_keras.h5"


# try:
#     model = load_model(model_path)
#     print("Model loaded successfully.")
#     print(model.summary())  # แสดงโครงสร้างของโมเดล
# except Exception as e:
#     print(f"Error loading model: {e}")


import tensorflow as tf

saved_model_path_h5 = "./saved_model_mobilefacenet/saved_model_mobilefacenet.h5"
saved_model_path_keras = "./saved_model_mobilefacenet/saved_model_mobilefacenet.keras"


# โหลดจากไฟล์ .keras
model_keras = tf.keras.models.load_model(saved_model_path_keras)
model_keras.summary()

# โหลดจากไฟล์ .h5
# model_h5 = tf.keras.models.load_model(saved_model_path_h5)
# model_h5.summary()



#  ตรวจสอบความเข้ากันได้ของเวอร์ชัน ดูว่าไฟล์ถูกสร้างด้วยเวอร์ชันอะไร:
# import h5py

# file_path = "D:/ayodia/top-panacea/Ayodia.PanaceaHS/Ayodia.PanaceaHS.WebAPI/Models/facenet_keras.h5"

# try:
#     with h5py.File(file_path, 'r') as f:
#         print("Keras version:", f.attrs.get('keras_version'))
#         print("Backend:", f.attrs.get('backend'))
# except Exception as e:
#     print(f"Error: {e}")



# แปลงไฟล์โมเดลเป็นรูปแบบใหม่ (SavedModel)
# from tensorflow.keras.models import load_model
# import tensorflow as tf

# # พาธของไฟล์โมเดล .h5
# old_model_path = "D:/ayodia/top-panacea/Ayodia.PanaceaHS/Ayodia.PanaceaHS.WebAPI/Models/facenet_keras.h5"

# # พาธที่จะบันทึกโมเดลใหม่
# new_model_path = "facenet_saved_model"

# # โหลดโมเดลที่สร้างด้วย Keras เก่า
# model = load_model(old_model_path)

# # บันทึกโมเดลในรูปแบบ SavedModel
# model.save(new_model_path)
# print("Model successfully converted to SavedModel format.")


