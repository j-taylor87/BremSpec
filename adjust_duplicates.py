import numpy as np

def adjust_duplicates(energy_array):
    """
    Adjusts duplicate values in an array by adding a small increment to ensure uniqueness.

    This function takes an array of energy values and checks for duplicates. If duplicates are found,
    it adds a small increment to each duplicate to make them unique while preserving the order.
    The resulting array is sorted and returned.

    Parameters:
    energy_array (array-like): Input array containing energy values.

    Returns:
    numpy.ndarray: A sorted array with adjusted values to ensure uniqueness.
    """
    unique_energy = {}
    increment = 0.0001  # Small increment to ensure uniqueness

    for i in range(len(energy_array)):
        if energy_array[i] in unique_energy:
            unique_energy[energy_array[i]] += 1
            energy_array[i] += unique_energy[energy_array[i]] * increment
        else:
            unique_energy[energy_array[i]] = 0

    return np.sort(energy_array)