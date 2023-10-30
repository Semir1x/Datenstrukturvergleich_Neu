import numpy as np
import tensorflow as tf
import sys
import time


class DataStructureComparison:
    def __init__(self, data):
        self.tensor_data = tf.convert_to_tensor(data)
        self.array_data = np.array(data)

    def measure_access_time(self, data_structure, index):
        start_time = time.time()
        _ = data_structure[index]
        return time.time() - start_time

    def compare_access_time(self, index=0):
        tensor_time = self.measure_access_time(self.tensor_data, index)
        array_time = self.measure_access_time(self.array_data, index)
        print(f"Zugriffszeit für Tensoren: {tensor_time:.6f} Sekunden")
        print(f"Zugriffszeit für Arrays: {array_time:.6f} Sekunden")

    def compare_memory_usage(self):
        tensor_memory = sys.getsizeof(self.tensor_data)
        array_memory = sys.getsizeof(self.array_data)
        print(f"Speicherverbrauch für Tensoren: {tensor_memory} Bytes")
        print(f"Speicherverbrauch für Arrays: {array_memory} Bytes")

    def measure_operation_efficiency(self, data_structure):
        start_time = time.time()
        _ = [x * 2 for x in data_structure]  # Beispieloperation: Verdopplung jedes Elements
        return time.time() - start_time

    def compare_operation_efficiency(self):
        tensor_time = self.measure_operation_efficiency(self.tensor_data)
        array_time = self.measure_operation_efficiency(self.array_data)
        print(f"Operationseffizienz für Tensoren: {tensor_time:.6f} Sekunden")
        print(f"Operationseffizienz für Arrays: {array_time:.6f} Sekunden")



    def measure_complex_operation_efficiency(self, data_structure):
        start_time = time.time()

        # Anwendung der Sigmoid-Funktion
        if isinstance(data_structure, tf.Tensor):
            _ = tf.math.sigmoid(data_structure)
        else:
            _ = 1 / (1 + np.exp(-data_structure))

        return time.time() - start_time

    def compare_complex_operation_efficiency(self):
        tensor_time = self.measure_complex_operation_efficiency(self.tensor_data)
        array_time = self.measure_complex_operation_efficiency(self.array_data)
        print(f"Komplexe Operationseffizienz für Tensoren: {tensor_time:.6f} Sekunden")
        print(f"Komplexe Operationseffizienz für Arrays: {array_time:.6f} Sekunden")


if __name__ == "__main__":
    sample_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    comparison = DataStructureComparison(sample_data)
    comparison.compare_access_time()
    comparison.compare_memory_usage()
    comparison.compare_operation_efficiency()
    comparison.compare_complex_operation_efficiency()
    comparison.compare_simplicity()
