import numpy as np
from scipy.constants import speed_of_light
import streamlit as st

@st.fragment
def kramers_law(target_material, energy, tube_voltage, tube_voltage_max, tube_voltage_min, tube_current=None, tube_current_max=None, exposure_time=None, exposure_time_max=None, current_time_product=None, current_time_product_max=None):
    """
    Calculate the normalised Bremsstrahlung spectrum based on Kramers" law for a given target material and set of operational parameters.

    This function computes the Bremsstrahlung radiation spectrum for a target material characterized by its atomic number. It considers different modes of operation (manual and automatic) based on the provided parameters. The output is the normalised energy flux of the radiation for energies up to the applied tube voltage.

    Parameters:
    Z (int): Atomic number of the target material.
    energy (ndarray): Array of electron energies (in keV).
    tube_voltage (float): Applied voltage setting the maximum electron energy for the protocol (in kV).
    tube_voltage_max (float): Maximum voltage setting the maximum electron energy for the modality (in kV).
    tube_current (float, optional): Tube current in mA (for manual mode).
    exposure_time (float, optional): Exposure time in seconds (for manual mode).
    current_time_product (float, optional): Current-time product in mAs (for automatic mode).
    current_time_product_max (float, optional): Maximum current-time product in mAs (for the modality in automatic mode).

    Returns:
    tuple of ndarray: A tuple containing two ndarrays. The first array is the valid energies up to the tube voltage, 
                      and the second array is the corresponding normalised energy flux of the radiation.
    """

    if target_material == "W (Z=74)":
        Z = 74
    
    elif target_material == "Rh (Z=45)":
        Z = 45

    elif target_material == "Mo (Z=42)":
        Z = 42

    k_l = 1  # Empirical constant
    # Filter out energy values that are greater than the tube_voltage
    energy_valid = energy[energy <= tube_voltage]

    # Calculate energy flux
    if current_time_product is not None:
        energy_flux = (k_l * Z * current_time_product) / (2.0 * np.pi * speed_of_light) * (tube_voltage - energy_valid) 
        energy_flux_max = (k_l * Z * current_time_product_max) / (2.0 * np.pi * speed_of_light) * (tube_voltage_max - energy_valid) 
    else:
        energy_flux = (k_l * Z * tube_current * exposure_time / 1000) / (2.0 * np.pi * speed_of_light) * (tube_voltage - energy_valid)
        energy_flux_max = (k_l * Z * tube_current_max * exposure_time_max / 1000) / (2.0 * np.pi * speed_of_light) * (tube_voltage_max - energy_valid)

    # Normalise energy flux
    energy_flux_normalised = energy_flux / np.max(energy_flux_max) 

    # Scale normalised energy flux relative to Tungsten, to show decreased X-ray production for other targets
    energy_flux_normalised = energy_flux_normalised* Z/74

    return energy_valid, energy_flux_normalised