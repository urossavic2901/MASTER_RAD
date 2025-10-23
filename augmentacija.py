import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import random
from tqdm import tqdm


input_root = "/content/drive/MyDrive/CCSN/dataset"
output_root = "/content/drive/MyDrive/CCSN/dataset_augmented"


IMG_SIZE = 256
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1),
])


def augment_and_save(image_path, save_dir, num_augmented):
    os.makedirs(save_dir, exist_ok=True)

    img = keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    for i in range(num_augmented):
        augmented = data_augmentation(img_array, training=True)
        out_img = tf.keras.utils.array_to_img(augmented[0])
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(save_dir, f"{base_name}_aug_{i+1}_{random.randint(10000,99999)}.jpg")
        out_img.save(out_path, "JPEG", quality=95)


for class_name in os.listdir(input_root):
    class_input_dir = os.path.join(input_root, class_name)
    class_output_dir = os.path.join(output_root, class_name)

    if not os.path.isdir(class_input_dir):
        continue

    os.makedirs(class_output_dir, exist_ok=True)

    existing_images = len([f for f in os.listdir(class_input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    target_total = 3000
    to_generate = max(0, target_total - existing_images)

    print(f" Klasa: {class_name}")
    print(f" - Postojeće slike: {existing_images}")
    print(f" - Potrebno generisati: {to_generate}")

    if to_generate == 0:
        print(" - Već ima dovoljno slika, preskačem.")
        continue

    
    image_files = [os.path.join(class_input_dir, f)
                   for f in os.listdir(class_input_dir)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    # Generiši nasumične slike dok ne dostigne cilj
    for i in tqdm(range(to_generate), desc=f"Generišem za {class_name}"):
        img_path = random.choice(image_files)
        augment_and_save(img_path, class_output_dir, num_augmented=1)

    print(f" Završeno za klasu {class_name}: generisano {to_generate} novih slika.")

print("Augmentacija završena! Sve slike su u:", output_root)
