from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf

def prepare_data():
    # Daten laden
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalisierung der Pixelwerte
    train_images_array = train_images.astype('float32') / 255
    test_images_array = test_images.astype('float32') / 255

    # One-Hot-Encoding der Labels
    train_labels_array = to_categorical(train_labels)
    test_labels_array = to_categorical(test_labels)

    # Aufteilen der Trainingsdaten in Trainings- und Validierungsdaten
    val_images_array = train_images_array[:5000]
    val_labels_array = train_labels_array[:5000]
    train_images_array = train_images_array[5000:]
    train_labels_array = train_labels_array[5000:]

    # Datenvermehrung
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )

    datagen.fit(train_images_array)

    # Konvertieren von Arrays zu Tensoren
    train_images_tensor = tf.convert_to_tensor(train_images_array)
    train_labels_tensor = tf.convert_to_tensor(train_labels_array)
    val_images_tensor = tf.convert_to_tensor(val_images_array)
    val_labels_tensor = tf.convert_to_tensor(val_labels_array)
    test_images_tensor = tf.convert_to_tensor(test_images_array)
    test_labels_tensor = tf.convert_to_tensor(test_labels_array)

    return (train_images_array, train_labels_array, val_images_array, val_labels_array, test_images_array, test_labels_array,
            train_images_tensor, train_labels_tensor, val_images_tensor, val_labels_tensor, test_images_tensor, test_labels_tensor, datagen)

if __name__ == "__main__":
    (train_images_array, train_labels_array, val_images_array, val_labels_array, test_images_array, test_labels_array,
     train_images_tensor, train_labels_tensor, val_images_tensor, val_labels_tensor, test_images_tensor, test_labels_tensor, datagen) = prepare_data()

    print(f"Trainingsdaten (Array): {train_images_array.shape}, {train_labels_array.shape}")
    print(f"Validierungsdaten (Array): {val_images_array.shape}, {val_labels_array.shape}")
    print(f"Testdaten (Array): {test_images_array.shape}, {test_labels_array.shape}")
    print(f"Trainingsdaten (Tensor): {train_images_tensor.shape}, {train_labels_tensor.shape}")
    print(f"Validierungsdaten (Tensor): {val_images_tensor.shape}, {val_labels_tensor.shape}")
    print(f"Testdaten (Tensor): {test_images_tensor.shape}, {test_labels_tensor.shape}")
