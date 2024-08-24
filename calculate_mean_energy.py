import numpy as np
import streamlit as st

@st.fragment
def calculate_mean_energy(energy_valid, energy_flux_normalised_filtered):

    # Calculate the total flux
    total_flux = np.trapz(energy_flux_normalised_filtered, energy_valid)

    # Calculate the weighted mean of the energy values
    mean_energy = np.trapz(energy_flux_normalised_filtered * energy_valid, energy_valid) / total_flux

    return mean_energy