import tensorflow as tf
import os

# ตั้งค่าพาธไปยังไฟล์ Checkpoint
meta_path = './src/MobileFaceNet_TF/arch/pretrained_model/MobileFaceNet_TF.ckpt.meta'  # พาธไฟล์ .meta
ckpt_path = './src/MobileFaceNet_TF/arch/pretrained_model/MobileFaceNet_TF.ckpt'      # พาธไฟล์ .ckpt
saved_model_dir = './saved_model_mobilefacenet'  # พาธสำหรับบันทึกโมเดลแบบ SavedModel
h5_model_path = './MobileFaceNet.h5'            # พาธสำหรับบันทึกโมเดลแบบ .h5

# ตรวจสอบว่าไฟล์ Checkpoint มีอยู่หรือไม่
if not os.path.exists(meta_path) or not os.path.exists(ckpt_path + '.index'):
    raise FileNotFoundError("Checkpoint files not found! Please check the paths.")

# ปิด Eager Execution
tf.compat.v1.disable_eager_execution()

# ลองโหลด Checkpoint
tf.compat.v1.reset_default_graph()
try:
    with tf.compat.v1.Session() as sess:
        # โหลด Graph
        saver = tf.compat.v1.train.import_meta_graph(meta_path, clear_devices=True)
        saver.restore(sess, ckpt_path)
        print("Checkpoint restored successfully!")

        # เอาเฉพาะ Inference Graph โดยลบ Node Training และ Gradient
        inference_graph_def = tf.compat.v1.graph_util.remove_training_nodes(
            tf.compat.v1.get_default_graph().as_graph_def()
        )

        # ตรวจสอบ Tensor ที่มีอยู่ใน Graph หลังการแก้ไข
        print("\nAvailable tensors in the graph after removing training nodes:")
        for node in inference_graph_def.node:
            print(node.name)

        # ระบุ Tensor สำหรับ Input และ Output
        input_tensor_name = "input_image:0"   # ชื่อ Tensor สำหรับอินพุต
        output_tensor_name = "embeddings:0"  # ชื่อ Tensor สำหรับเอาต์พุต

        input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(input_tensor_name)
        output_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(output_tensor_name)

        # บันทึกโมเดลในรูปแบบ SavedModel
        print("\nSaving model as TensorFlow SavedModel...")
        tf.saved_model.simple_save(
            sess,
            saved_model_dir,
            inputs={"input_image": input_tensor},
            outputs={"embeddings": output_tensor},
        )
        print(f"Model successfully saved in SavedModel format at: {saved_model_dir}")

        # สร้างโมเดลใหม่ใน TensorFlow 2.x และบันทึกเป็น HDF5
        print("\nConverting model to Keras HDF5 format...")
        model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
        model.save(h5_model_path, save_format='h5')
        print(f"Model successfully saved in HDF5 format at: {h5_model_path}")

except ValueError as e:
    print(f"Error restoring graph or tensors: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
