# Load in NIST data and interpolate to get more energy and mass attenuation coefficient values

import pandas as pd
# from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
import numpy as np
import os

def load_mass_attenuation_data(file_path):
    """
    Load mass attenuation coefficient data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing energy and mass attenuation coefficient data.

    Returns:
    tuple:
      energies (np.array): Array of energy values.
      mass_atten_coeffs (np.array): Array of mass attenuation coefficients corresponding to the energy values.
    """
    df = pd.read_csv(file_path, delimiter='\t')
    print("Interpolating data in file:",file_path)
    df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace from column names
    energy_keV = df['Energy_MeV'].to_numpy() * 1000  # Convert MeV to keV
    mass_atten_coeff_cm2_g = df['mass_atten_coeff_cm2_g'].to_numpy()
    # print(energy_keV)
    
    return energy_keV, mass_atten_coeff_cm2_g

def adjust_duplicates(energy_array):
    """
    Adjusts duplicate values in an array by adding a small increment to ensure uniqueness. 
    This avoids plotting errors in the main script.

    This function takes an array of energy values and checks for duplicates. If duplicates are found,
    it adds a small increment to each duplicate to make them unique while preserving the order.
    The resulting array is sorted and returned.

    Parameters:
    energy_array (array-like): Input array containing energy values.

    Returns:
    numpy.ndarray: A sorted array with adjusted values to ensure uniqueness.
    """
    unique_energy = {}
    increment = 0.000001  # Small increment to ensure uniqueness, to avoid plotting errors later

    for i in range(len(energy_array)):
        if energy_array[i] in unique_energy:
            unique_energy[energy_array[i]] += 1
            energy_array[i] += unique_energy[energy_array[i]] * increment
        else:
            unique_energy[energy_array[i]] = 0

    return np.sort(energy_array)

def interpolate_and_save(file_path, base_energy_array, output_file_path):
    """
    Interpolate the mass attenuation coefficients to match the base energy array and save the results to a new CSV file.

    Parameters:
    file_path (str): Path to the original CSV file containing energy and mass attenuation coefficient data.
    base_energy_array (np.array): Array of energy values (in keV) to which the mass attenuation coefficients will be interpolated.
    output_file_path (str): Path to save the interpolated data CSV file.
    """
    energy_keV, mass_atten_coeff_cm2_g = load_mass_attenuation_data(file_path)
    
    # Adjust duplicates in energy array
    energy_keV = adjust_duplicates(energy_keV)
    
    # Interpolation function
    # interpolate_func = interp1d(energy_keV, mass_atten_coeff_cm2_g, kind="linear", fill_value="extrapolate")
    interpolate_func = PchipInterpolator(energy_keV, mass_atten_coeff_cm2_g)
    interpolated_mass_atten_coeffs = interpolate_func(base_energy_array)
    
    # Save to new CSV
    df_interpolated = pd.DataFrame({
        'energy_keV': base_energy_array,
        'mass_atten_coeff_cm2_g': interpolated_mass_atten_coeffs
    })
    df_interpolated.to_csv(output_file_path, index=False)

data_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Base energy array (example)
base_energy_array = np.linspace(0, 150, 10000)  # 0 keV to 150 keV in n steps

# Interpolate and save for each CSV file in the data directory
for file_name in os.listdir(data_dir):
    if file_name.endswith(".csv") and not file_name.startswith("interpolated_"):
        input_file_path = os.path.join(data_dir, file_name)
        output_file_path = os.path.join(data_dir, f"interpolated_{file_name}")
        interpolate_and_save(input_file_path, base_energy_array, output_file_path)
