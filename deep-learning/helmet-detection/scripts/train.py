from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators():
    # مسیر پوشه‌های داده
    train_dir = 'data/helmet_detection/train'
    image_size = (224, 224)

    # تنظیمات ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # ایجاد train_generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    # ایجاد validation_generator
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

