import numpy as np
import streamlit as st

@st.fragment
def calculate_median_energy(energy_valid, energy_flux_normalised_filtered):

    # Calculate the cumulative sum of the energy fluxes
    cumulative_energy_flux = np.cumsum(energy_flux_normalised_filtered * np.diff(energy_valid, prepend=0))

    # Normalise by the total AUC
    normalised_cumulative_energy_flux = cumulative_energy_flux / np.trapz(energy_flux_normalised_filtered, energy_valid)

    # Find the index for median energy
    indices = np.where(normalised_cumulative_energy_flux >= 0.5)[0]
    if len(indices) > 0:
        median_index = indices[0]
        median_energy_at_50pct_auc = energy_valid[median_index]

        return median_energy_at_50pct_auc
    else:
        # Handle the case where no median is found
        return None  # Return None when median is not found