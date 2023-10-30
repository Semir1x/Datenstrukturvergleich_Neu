import tensorflow as tf
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

def train_and_evaluate(model, train_images, train_labels, val_images, val_labels, test_images, test_labels, datagen):
    # Kompilieren des Modells
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks definieren
    checkpoint = ModelCheckpoint('best_model.tf', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

    # Training des Modells mit Datenvermehrung
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=64),
                        epochs=1,
                        validation_data=(val_images, val_labels),
                        callbacks=[checkpoint, early_stopping])

    # Evaluieren des Modells mit den Testdaten
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    return history

def plot_training_results(history):


    # Genauigkeit
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')

    # Verlust darstellen
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss over epochs')

    plt.show()

if __name__ == "__main__":
    # Testzwecke
    from data_preparation import prepare_data
    from cnn_model import create_cnn_model

    (train_images_array, train_labels_array, val_images_array, val_labels_array, test_images_array, test_labels_array,
     train_images_tensor, train_labels_tensor, val_images_tensor, val_labels_tensor, test_images_tensor, test_labels_tensor, datagen) = prepare_data()
    model = create_cnn_model()
    history = train_and_evaluate(model, train_images_array, train_labels_array, val_images_array, val_labels_array, test_images_array,
                                 test_labels_array, datagen)

    # Visualisieren der Ergebnisse
    plot_training_results(history)
