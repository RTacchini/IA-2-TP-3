import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Configuración general
img_height, img_width = 64, 64  # Redimensionar imágenes
batch_size = 32
epochs = 100  # Puedes ajustar según necesidad

# Preparar generador de imágenes desde carpetas
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizar
    validation_split=0.2
)

# Solo clases válidas
CLASES_VALIDAS = ['up', 'down', 'right']

train_generator = datagen.flow_from_directory(
    'images',
    classes=CLASES_VALIDAS,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    'images',
    classes=CLASES_VALIDAS,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation',
    shuffle=True
)

# Crear el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 clases: 0 = up, 1 = down, 2 = right
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Guardar el modelo
model.save("modelo_dino.h5")
