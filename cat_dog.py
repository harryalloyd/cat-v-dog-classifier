import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def remove_corrupt_images(folder_path):
    """Removes corrupt images in the given folder by trying to open each file with Pillow."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Raises an exception if the file is corrupt
            except Exception:
                print(f"Removing corrupt file: {file_path}")
                os.remove(file_path)

# 1. Remove corrupt images from both 'Cat' and 'Dog' subfolders
remove_corrupt_images("PetImages/Cat")
remove_corrupt_images("PetImages/Dog")

# 2. Basic parameters
img_height = 180
img_width = 180
batch_size = 32

# 3. Data Augmentation: helps the model generalize better
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# 4. Create training and validation datasets from "PetImages"
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
).apply(tf.data.experimental.ignore_errors())

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 5. Prefetch and cache to improve loading performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 6. Transfer Learning: Use MobileNetV2 as a feature extractor
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze the base model

# 7. Build the new model incorporating data augmentation and the base model
inputs = keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)  # Apply augmentation
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # Preprocess input for MobileNetV2
x = base_model(x, training=False)  # Extract features
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)  # Add dropout for regularization
outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary classification head
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 8. Train the model
epochs = 5
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)

# 9. Evaluate on the validation set
val_loss, val_acc = model.evaluate(val_ds)
print(f"Validation accuracy: {val_acc:.2f}")

# 10. Example: Predict on a single new image (replace 'dog1.jpg' with your own image)
new_img_path = "put_your_image_here.jpg"
img = keras.preprocessing.image.load_img(new_img_path, target_size=(img_height, img_width))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch of 1

prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("This looks like a Dog.")
else:
    print("This looks like a Cat.")
