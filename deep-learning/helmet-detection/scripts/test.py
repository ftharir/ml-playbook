from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# بارگذاری مدل
model = load_model('models/helmet_detection_model.h5')
# the saved file keeps only the weights (no optimizer state), so compile again before evaluating
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# تنظیمات داده‌های تست
test_dir = 'data/helmet_detection/test'
image_size = (224, 224)

# پیش‌پردازش داده‌های تست
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='binary')

# ارزیابی مدل
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // 32)
print(f'Test accuracy: {test_acc * 100:.2f}%')
