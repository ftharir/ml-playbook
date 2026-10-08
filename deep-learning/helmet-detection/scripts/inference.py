from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import sys

# بارگذاری مدل آموزش‌دیده
model = load_model('models/helmet_detection_model.h5')

# مسیر تصویر ورودی
image_path = sys.argv[1]  # usage: python scripts/inference.py path/to/image.png

# بررسی وجود فایل
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

# بارگذاری تصویر
image = cv2.imread(image_path)

# بررسی موفقیت بارگذاری تصویر
if image is None:
    raise ValueError(f"Failed to load image. Please check the file format or path: {image_path}")

# تغییر اندازه تصویر
image = cv2.resize(image, (224, 224))

# آماده‌سازی تصویر برای پیش‌بینی
image = np.expand_dims(image, axis=0)
image = image / 255.0  # نرمال‌سازی تصویر

# پیش‌بینی
prediction = model.predict(image)
if prediction > 0.5:
    print("The worker is wearing a helmet.")
else:
    print("The worker is not wearing a helmet.")
