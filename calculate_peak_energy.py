import numpy as np
import streamlit as st

@st.fragment
def calculate_peak_energy(energy_valid, energy_flux_normalised_filtered):
    """
    Calculate the energy corresponding to the highest energy flux.

    Parameters:
    energy_valid (np.array): Array of energy values.
    energy_flux_normalised_filtered (np.array): Array of normalized energy flux values.

    Returns:
    float: Energy corresponding to the highest flux.
    """
    # Find the index of the maximum energy flux
    peak_index = np.argmax(energy_flux_normalised_filtered)
    
    # Get the corresponding energy value
    peak_energy = energy_valid[peak_index]
    
    return peak_energy