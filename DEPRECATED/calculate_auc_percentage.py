import numpy as np
import streamlit as st

@st.fragment
def calculate_auc_percentage(energy_flux_normalised_filtered, energy_valid, energy_lower_bound, energy_upper_bound, tube_voltage_max):
    """
    Calculate the AUC percentage for a filtered energy spectrum within a specified energy range.

    Parameters:
    energy_flux_normalised_filtered (numpy.ndarray): normalised filtered energy flux.
    energy_valid (numpy.ndarray): Valid energy values.
    energy_lower_bound (float): Lower bound of the energy range of interest.
    energy_upper_bound (float): Upper bound of the energy range of interest.
    tube_voltage_max (float): Maximum tube voltage.

    Returns:
    auc_percentage (float): AUC percentage within the specified energy range.
    """
    # Indices for the energy range of interest
    lower_index = np.searchsorted(energy_valid, energy_lower_bound, side="left")
    upper_index = np.searchsorted(energy_valid, energy_upper_bound, side="right")

    # Calculate the AUC for the unfiltered spectrum at maximum technique factor values
    auc_unfiltered = 0.5 * tube_voltage_max * 1.0

    # Calculate AUC within the specified energy range
    auc = np.trapz(energy_flux_normalised_filtered[lower_index:upper_index], energy_valid[lower_index:upper_index])

    # Calculate AUC percentage
    auc_percentage = (auc / auc_unfiltered) * 100

    return auc_percentage