from data_preparation import prepare_data
from cnn_model import create_cnn_model
from train_model import train_and_evaluate, plot_training_results
from CompareStructures import DataStructureComparison

def main():
    # Daten laden und vorbereiten
    (train_images_array, train_labels_array, val_images_array, val_labels_array, test_images_array, test_labels_array,
     train_images_tensor, train_labels_tensor, val_images_tensor, val_labels_tensor, test_images_tensor, test_labels_tensor, datagen) = prepare_data()

    # CNN-Modell erstellen
    model = create_cnn_model(input_shape=(32, 32, 3))

    # Modell trainieren und bewerten mit Arrays
    print("Training und Bewertung mit Arrays:")
    history_array = train_and_evaluate(model, train_images_array, train_labels_array, val_images_array, val_labels_array, test_images_array, test_labels_array,
                                       datagen)
    plot_training_results(history_array)

    # Modell trainieren und bewerten mit Tensoren
    print("Training und Bewertung mit Tensoren:")
    history_tensor = train_and_evaluate(model, train_images_tensor, train_labels_tensor, val_images_tensor, val_labels_tensor, test_images_tensor,
                                        test_labels_tensor, datagen)
    plot_training_results(history_tensor)

    # Vergleich zwischen Tensoren und Arrays
    sample_data = [train_images_array[0], train_images_array[1], train_images_array[2]]  # Nehmen Sie die ersten 3 Bilder als Beispiel
    comparison = DataStructureComparison(sample_data)
    comparison.compare_access_time()
    comparison.compare_memory_usage()
    comparison.compare_operation_efficiency()
    comparison.compare_complex_operation_efficiency()
if __name__ == "__main__":
    main()
