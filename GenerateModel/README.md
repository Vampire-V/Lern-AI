voice-command-recognition/
│
├── data/                             # เก็บไฟล์ข้อมูลเสียง
│   ├── sawasdee/                     # เสียงคำว่า "สวัสดี"
│   └── other_sounds/                 # เสียงอื่นๆ
│
├── models/                           # โฟลเดอร์เก็บโมเดลที่ฝึกแล้ว
│   └── sawasdee_model.h5             # โมเดลที่ฝึกแล้ว
│
├── notebooks/                        # สำหรับเก็บไฟล์ Jupyter Notebooks (ถ้ามี)
│   └── training.ipynb                # Notebook สำหรับการทดสอบและฝึกโมเดล
│
├── src/                              # โฟลเดอร์สำหรับ source code หลัก
│   ├── preprocessing.py              # โมดูลสำหรับการเตรียมข้อมูล (Preprocessing)
│   ├── model_training.py             # โมดูลสำหรับการสร้างและฝึกโมเดล
│   ├── predict.py                    # โมดูลสำหรับโหลดโมเดลและทำนายเสียง
│   └── utils.py                      # ฟังก์ชันเสริมต่างๆ ที่ใช้ในโปรเจกต์
│
├── saved_features/                   # เก็บข้อมูลเสียงที่ถูกแปลงเป็น features (เช่น MFCCs)
│   ├── file1.npy                     # ไฟล์เสียงที่แปลงแล้ว
│   └── file2.npy
│
├── requirements.txt                  # รายการ dependencies (librosa, tensorflow, numpy ฯลฯ)
└── README.md                         # คำอธิบายโปรเจกต์




1.ติดตั้ง Dependencies:
    - สร้าง virtual environment และติดตั้ง dependencies ที่ระบุใน requirements.txt:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

2.เตรียมข้อมูลเสียง:
    - เพิ่มไฟล์เสียงในโฟลเดอร์ data/hello_volkswagen/ และ data/other_sounds/ ตามที่กำหนด
3.Preprocessing ข้อมูล:
    - ใช้โมดูล preprocessing.py เพื่อแปลงไฟล์เสียงทั้งหมดใน data/ ให้เป็นคุณลักษณะเสียง เช่น MFCCs และบันทึกผลลัพธ์ใน saved_features/
4.ฝึกโมเดล:
    -ใช้ model_training.py ในการฝึกโมเดล โดยโมเดลที่ฝึกแล้วจะถูกบันทึกในโฟลเดอร์ models/
5.ทำนายเสียง:
    - ใช้ predict.py เพื่อทำนายไฟล์เสียงใหม่โดยใช้โมเดลที่ฝึกแล้ว