import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# === Configuration ===
dataset_path = 'dataset'
normal_path = os.path.join(dataset_path, 'normal')
diabetic_retinopathy_path = os.path.join(dataset_path, 'diabetic_retinopathy')
glaucoma_path = os.path.join(dataset_path, 'glaucoma')
cataract_path = os.path.join(dataset_path, 'cataract')
img_size = (224, 224)
batch_size = 32
epochs = 10

# Create folder to save model
model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)
model_save_path = os.path.join(model_dir,'saved_models', 'eye_disease_model.h5')

# === Data Preparation ===
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === MobileNetV2 Model ===
base_model = MobileNetV2(input_shape=(img_size[0], img_size[1], 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# === Compile Model ===
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Train Model ===
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# === Save Model ===
model.save(model_save_path)
abs_path = os.path.abspath(model_save_path)
print(f"✅ Model saved to: {abs_path}")

# === Evaluate Model ===
val_loss, val_acc = model.evaluate(val_data)
print(f"✅ Validation Accuracy: {val_acc * 100:.2f}%")

# === Plot Training History ===
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.grid(True)
plt.show()
