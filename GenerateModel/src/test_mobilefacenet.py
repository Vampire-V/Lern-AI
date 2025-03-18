from MobileFaceNet_TF.nets.MobileFaceNetCustom import mobilenet_v2
import tensorflow as tf

# สร้างอินพุตจำลอง
inputs = tf.keras.Input(shape=(112, 112, 3), name="input_image")

# สร้างโมเดล MobileFaceNet
try:
    net, end_points = mobilenet_v2(inputs)

    # แปลงเป็น Keras Model
    model = tf.keras.Model(inputs=inputs, outputs=net, name="MobileFaceNet")

    # เรียก summary() บนโมเดล
    print("MobileFaceNet Model built successfully!")
    model.summary()
except Exception as e:
    print(f"Error building MobileFaceNet Model: {e}")