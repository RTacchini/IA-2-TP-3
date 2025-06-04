import subprocess
try:
    import tensorflow as tf
except ImportError as err:
    subprocess.check_call(['pip', 'install', 'tensorflow'])
    subprocess.check_call(['pip', 'install', 'Pillow'])
    import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import random
import matplotlib.pyplot as plt
import json 

# Rutas
source_dir = "images"
train_dir = os.path.join(source_dir, "train")
test_dir = os.path.join(source_dir, "test")

# Clases
classes = ["up", "down", "right"]

# Crear carpetas necesarias
for base_dir in [train_dir, test_dir]:
    for class_name in classes:
        os.makedirs(os.path.join(base_dir, class_name), exist_ok=True)

# Parámetros
train_ratio = 0.8
batch_size = 32
image_size = (128, 200)
input_shape = image_size + (1,)  # Grayscale = 1 canal

# Cargar y normalizar imágenes en escala de grises
def load_and_preprocess_image(file_path, target_size):
    img = load_img(file_path, color_mode='grayscale', target_size=target_size)
    img_array = img_to_array(img)
    return img_array

# Distribuir imágenes en train/test
for class_name in classes:
    source_class_dir = os.path.join(source_dir, class_name)
    images = os.listdir(source_class_dir)
    random.shuffle(images)
    num_train = int(len(images) * train_ratio)

    for i, img_name in enumerate(images):
        src = os.path.join(source_class_dir, img_name)
        dst_base = train_dir if i < num_train else test_dir
        dst = os.path.join(dst_base, class_name, img_name)
        img_array = load_and_preprocess_image(src, image_size)

        # Mostrar una imagen para verificar
        if i == 0 and dst_base == train_dir:  # Solo muestra una vez por clase
            plt.imshow(img_array.squeeze(), cmap='gray')
            plt.title(f"Clase: {class_name} - Imagen: {img_name}")
            plt.axis('off')
            plt.show()

        tf.keras.preprocessing.image.save_img(dst, img_array, scale=False)

# Generadores
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    color_mode='grayscale',
    shuffle=True
)

# Mostrar e imprimir el mapeo
print("Índices de clase asignados:", train_generator.class_indices)

# Guardar el mapeo a un archivo JSON
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

validation_generator = datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    color_mode='grayscale',
    shuffle=True
)

# Modelo convolucional
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compilación
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Guardado del modelo
model.save('modelo_dino.h5')
