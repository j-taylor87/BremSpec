import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["font.sans-serif"] = "Tahoma"
from scipy.interpolate import interp1d
from scipy.constants import speed_of_light

def kramers_law(Z, energy, tube_voltage, tube_voltage_max, tube_current=None, tube_current_max=None, exposure_time=None, exposure_time_max=None, current_time_product=None, current_time_product_max=None):
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

    # normalise energy flux
    energy_flux_normalised = energy_flux / np.max(energy_flux_max)
    
    return energy_valid, energy_flux_normalised

def relative_attenuation_mass_coeff(energy, density, filter_thickness, mass_atten_coeff, tube_voltage):
    """
    Calculate the relative attenuation of radiation through a material based on its mass attenuation coefficient.

    This function computes the relative attenuation of X-rays or gamma rays as they pass through 
    a given material, which is useful in applications like medical imaging and radiation shielding.

    Parameters:
    energy (ndarray): An array of photon energies (in keV) for which the attenuation is to be calculated.
    density (float): The density of the material (in g/cm³).
    thickness (float): The thickness of the material through which the radiation is passing (in mm).
    mass_atten_coeff (ndarray): An array of mass attenuation coefficients (in cm²/g) corresponding to the energies in the "energy" array.
        These coefficients can be obtained from NIST"s XCOM database.
    tube_voltage (float): The maximum voltage setting (in kV) defining the upper limit of the energy range for calculations.

    Returns:
    ndarray
        An array of relative attenuation values corresponding to each energy value. This represents the 
        fraction of radiation intensity that is not attenuated by the material.

    Note:
    - The NIST XCOM calculator (https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html) can be used 
      to obtain mass attenuation coefficients for different materials and energies.
    - The thickness of the material is converted from mm to cm within the function for calculation purposes.
    """
    mass_atten_coeff_valid = mass_atten_coeff[energy <= tube_voltage]
    attenuation_relative = np.exp(-mass_atten_coeff_valid * filter_thickness / 10 * density)  # /10 to convert thickness from mm to cm

    return attenuation_relative

def add_characteristic_peaks(energy, energy_flux_normalised_filtered, energy_char, flux_peaks, tube_voltage):
    """
    Integrate characteristic X-ray peaks into an existing normalised Bremsstrahlung spectrum.

    This function adds specified characteristic peaks to an existing spectrum, normalises their intensities 
    relative to the spectrum"s maximum intensity and selected kV, and then sorts the combined energy and intensity arrays 
    for consistent plotting. It is particularly useful for visualising the complete spectrum including both 
    the Bremsstrahlung continuum and characteristic radiation peaks.

    Parameters:
    energy (ndarray): An array of energies in the existing spectrum.
    energy_flux_normalised_filtered (ndarray): An array of normalised energy flux values corresponding to "energy".
    energy_char (ndarray): An array of energies where characteristic peaks occur.
    flux_peaks (ndarray): An array of flux values for each characteristic peak.

    Returns:
    tuple of ndarray: A tuple containing two sorted ndarrays. The first array is the combined energy values (including characteristic peaks),
                      and the second array is the corresponding combined and normalised energy flux values.
    """
    # Filter out energies and their corresponding flux values above the tube_voltage
    energy_valid = energy[energy <= tube_voltage]
    flux_valid = energy_flux_normalised_filtered[energy <= tube_voltage]

    # Filter out characteristic peak energies above the tube_voltage
    peak_energies_valid = [en for en in energy_char if en <= tube_voltage]
    peak_fluxes_valid = [flux_peaks[i] for i, e in enumerate(energy_char) if e <= tube_voltage]

    # Normalise and adjust peak fluxes
    peak_fluxes_normalised = [flux * max(flux_valid) for flux in peak_fluxes_valid]

    for i, peak_energy in enumerate(peak_energies_valid):
        # Find the closest intensity in the valid Bremsstrahlung spectrum for each peak
        closest_index = np.abs(energy_valid - peak_energy).argmin()
        closest_intensity = flux_valid[closest_index]

        # Ensure peak intensity is not less than the Bremsstrahlung intensity at that energy
        if peak_fluxes_normalised[i] < closest_intensity:
            peak_fluxes_normalised[i] = closest_intensity

    # Combine and sort the valid energy and intensity arrays
    energy_combined = np.append(energy_valid, peak_energies_valid)
    intensity_combined = np.append(flux_valid, peak_fluxes_normalised)
    sorted_indices = np.argsort(energy_combined)

    return energy_combined[sorted_indices], intensity_combined[sorted_indices]

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

def interpolate_mass_atten_coeff(base_energy_array, energy_array, mass_atten_coeff):
    """
    Interpolates the mass attenuation coefficients for a given array of base energies.

    This function takes an array of base energies and uses linear interpolation to
    find the corresponding mass attenuation coefficients based on a provided array
    of energies and their known mass attenuation coefficients. It handles duplicate
    energy values and extrapolates values for energies not in the provided range.

    Parameters:
    base_energy_array (numpy.ndarray): An array of energies for which the mass 
      attenuation coefficients need to be interpolated.
    energy_array (numpy.ndarray): The array of energies corresponding to the provided 
      mass attenuation coefficients. This array is used as the base for interpolation.
    mass_atten_coeff (numpy.ndarray): An array of known mass attenuation coefficients 
      corresponding to the energies in `energy_array`.

    Returns:
    numpy.ndarray: An array of interpolated mass attenuation coefficients corresponding 
      to the energies in `base_energy_array`.
    """
    # Adjust duplicate values in the energy array for accurate interpolation
    energy = adjust_duplicates(energy_array)

    # Create a linear interpolation function
    interpolate = interp1d(energy, mass_atten_coeff, kind="linear", fill_value="extrapolate")

    # Interpolate the mass attenuation coefficients
    mass_atten_coeff_interpolated = interpolate(base_energy_array)

    return mass_atten_coeff_interpolated

def filter_selection_and_input(base_energy_array, filter_number, filters, default=None):
    """
    Selects a filter material and thickness, and interpolates its mass attenuation coefficients.

    This function allows users to select a filter material from a given list and set its thickness using a slider. 
    It then interpolates the mass attenuation coefficients of the selected material to match a base energy array.
    The function handles different materials with specific energy arrays and mass attenuation coefficients retrieved from NIST's XCOM database.
    For more information, refer to: https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html

    Parameters:
    base_energy_array (np.array): An array of energy values (in keV) to which the mass attenuation coefficients will be interpolated.
    filter_number (int): The number of the filter (used for labeling in the user interface).
    filters (list): A list of filter materials available for selection.
    default (str, optional): The default filter material to be preselected. Default is None.

    Returns:
    tuple:
      interpolated_mass_attenuation (np.array): The interpolated mass attenuation coefficients.
      selected_filter (str): The selected filter material.
      filter_density (float): The density of the selected material (in g/cm^3).
      selected_thickness (float): The selected thickness of the filter (in mm).
    """
    
    # Use the default filter if provided, otherwise default to the first item in the list
    default_index = filters.index(default) if default else 0
    filter_material_selection = st.selectbox(f"Filter {filter_number} Material", filters, index=default_index, key=f"filter_material_{filter_number}")
  
    if filter_material_selection == "Al (Z=13)":
        density = 2.7  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Al", min_value=0.0, max_value=21.0, step=0.05, value=1.0, key=f"filter_{filter_number}_thickness_Al")
        energy_Al = np.array([1.0,1.5,1.56,1.56,2,2.51,4.01,5.52,7.02,8.53,10.03,11.54,13.04,14.55,16.05,17.56,19.06,20.57,22.07,23.58,25.08,26.59,28.09,29.6,31.1,32.61,34.11,35.62,37.12,38.63,40.13,41.64,43.14,44.65,46.15,47.66,49.16,50.67,52.17,53.68,55.18,56.69,58.19,59.7,61.2,62.71,64.21,65.72,67.22,68.73,70.23,71.74,73.24,74.75,76.25,77.76,79.26,80.77,82.27,83.78,85.28,86.79,88.29,89.8,91.3,92.81,94.31,95.82,97.32,98.83,100.3,101.8,103.3,104.8,106.4,107.9,109.4,110.9,112.4,113.9,115.4,116.9,118.4,119.9,121.4,122.9,124.4,125.9,127.4,128.9,130.4,131.9,133.4,134.9,136.4,138,139.5,141,142.5,144,145.5,147,148.5,150,130.4,131.9,133.4,134.9,136.4,138,139.5,141,142.5,144,145.5,147,148.5,150,200])
        mass_atten_coeff_Al = np.array([1183,400.2,360,3955,2261,1271,356.7,145.1,72.64,41.07,25.43,16.75,11.63,8.372,6.239,4.767,3.734,2.979,2.423,1.999,1.675,1.419,1.218,1.056,0.9255,0.8178,0.7294,0.6552,0.5934,0.5408,0.4963,0.4579,0.4251,0.3965,0.3717,0.35,0.331,0.3141,0.2993,0.286,0.2742,0.2636,0.2542,0.2456,0.2378,0.2308,0.2244,0.2185,0.2132,0.2083,0.2038,0.1997,0.1958,0.1923,0.189,0.1859,0.1831,0.1804,0.1779,0.1755,0.1733,0.1713,0.1693,0.1675,0.1657,0.1641,0.1625,0.161,0.1596,0.1582,0.1569,0.1557,0.1545,0.1533,0.1522,0.1512,0.1502,0.1492,0.1483,0.1474,0.1465,0.1456,0.1448,0.144,0.1432,0.1425,0.1418,0.141,0.1404,0.1397,0.139,0.1384,0.1378,0.1372,0.1366,0.136,0.1354,0.1348,0.1343,0.1338,0.1332,0.1327,0.1322,0.1317,0.139,0.1384,0.1378,0.1372,0.1366,0.136,0.1354,0.1348,0.1343,0.1338,0.1332,0.1327,0.1322,0.1317,0.1188])
        #mass_atten_coeff = np.interp(base_energy_array, energy_Al, mass_atten_coeff_Al)
        mass_atten_coeff = interpolate_mass_atten_coeff(base_energy_array, energy_Al, mass_atten_coeff_Al)

    elif filter_material_selection == "Cu (Z=29)":
        density = 8.96  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Cu", min_value=0.0, max_value=1.0, step=0.01,value=0.0, key=f"filter_{filter_number}_thickness_Cu")
        energy_Cu = np.array([1.0,1.096,1.096,1.5,2,2.51,4.01,5.52,7.02,8.53,10.03,11.54,13.04,14.55,16.05,17.56,19.06,20.57,22.07,23.58,25.08,26.59,28.09,29.6,31.1,32.61,34.11,35.62,37.12,38.63,40.13,41.64,43.14,44.65,46.15,47.66,49.16,50.67,52.17,53.68,55.18,56.69,58.19,59.7,61.2,62.71,64.21,65.72,67.22,68.73,70.23,71.74,73.24,74.75,76.25,77.76,79.26,80.77,82.27,83.78,85.28,86.79,88.29,89.8,91.3,92.81,94.31,95.82,97.32,98.83,100.3,101.8,103.3,104.8,106.4,107.9,109.4,110.9,112.4,113.9,115.4,116.9,118.4,119.9,121.4,122.9,124.4,125.9,127.4,128.9,130.4,131.9,133.4,134.9,136.4,138,139.5,141,142.5,144,145.5,147,148.5,150,127.4,128.9,130.4,131.9,133.4,134.9,136.4,138,139.5,141,142.5,144,145.5,147,148.5,150,200])
        mass_atten_coeff_Cu = np.array([10570,8240,9340,4413,2149,1187,341.6,142.4,72.98,42.32,213,149.2,107.5,79.57,60.77,47.5,37.92,30.7,25.23,20.95,17.6,14.91,12.76,10.99,9.543,8.335,7.33,6.476,5.756,5.136,4.606,4.146,3.748,3.398,3.094,2.824,2.587,2.375,2.188,2.02,1.871,1.736,1.615,1.505,1.406,1.316,1.235,1.16,1.092,1.03,0.9727,0.9201,0.872,0.8274,0.7865,0.7484,0.7134,0.6806,0.6505,0.6222,0.5961,0.5716,0.5488,0.5274,0.5075,0.4888,0.4713,0.4548,0.4394,0.4248,0.4111,0.3981,0.386,0.3745,0.3636,0.3533,0.3436,0.3343,0.3256,0.3173,0.3094,0.3019,0.2948,0.288,0.2815,0.2754,0.2695,0.2639,0.2586,0.2534,0.2485,0.2439,0.2394,0.2351,0.231,0.227,0.2233,0.2196,0.2161,0.2128,0.2096,0.2064,0.2035,0.2006,0.2586,0.2534,0.2485,0.2439,0.2394,0.2351,0.231,0.227,0.2233,0.2196,0.2161,0.2128,0.2096,0.2064,0.2035,0.2006,0.1437])
        #mass_atten_coeff = np.interp(base_energy_array, energy_Cu, mass_atten_coeff_Cu)
        
        energy_Cu = adjust_duplicates(energy_Cu)
        interpolate_Cu = interp1d(energy_Cu, mass_atten_coeff_Cu, kind="linear", fill_value="extrapolate")
        mass_atten_coeff = interpolate_Cu(base_energy_array)

    elif filter_material_selection == "Mo (Z=42)":
        density = 10.2  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Mo", min_value=0.0, max_value=0.1, step=1E-6, value=0.0, key=f"filter_{filter_number}_thickness_Mo")
        energy_Mo = np.array([1,1.5,2,2.51,2.52,2.52,2.572,2.625,2.625,2.743,2.865,2.865,3,4,4.01,5,5.52,6,7.02,8,8.53,10,10.03,11.54,13.04,14.55,15,16.05,17.56,19.06,20,20,20,20.57,22.07,23.58,25.08,26.59,28.09,29.6,30,31.1,32.61,34.11,35.62,37.12,38.63,40,40.13,41.64,43.14,44.65,46.15,47.66,49.16,50,50.67,52.17,53.68,55.18,56.69,58.19,59.7,60,61.2,62.71,64.21,65.72,67.22,68.73,70.23,71.74,73.24,74.75,76.25,77.76,79.26,80,80.77,82.27,83.78,85.28,86.79,88.29,89.8,91.3,92.81,94.31,95.82,97.32,98.83,100,100.3,101.8,103.3,104.8,106.4,107.9,109.4,110.9,112.4,113.9,115.4,116.9,118.4,119.9,121.4,122.9,124.4,125.9,127.4,128.9,130.4,131.9,133.4,134.9,136.4,138,139.5,141,142.5,144,145.5,147,148.5,150,200])
        mass_atten_coeff_Mo = np.array([4943,1925,959.6,547,541.6,1979,1861,1750,2433,2184,1961,2243,2011,970.6,964.4,545,419.9,337.2,221.7,156.6,131.6,85.76,85.06,58.08,41.7,30.99,28.54,23.68,18.51,14.84,13.08,79.55,79.55,78.87,72.23,62.05,51.63,42.43,35.05,29.35,28.1,25.14,21.93,19.42,17.37,15.65,14.15,12.94,12.83,11.64,10.58,9.632,8.792,8.041,7.378,7.04,6.785,6.261,5.79,5.37,4.988,4.645,4.332,4.274,4.049,3.789,3.553,3.335,3.137,2.953,2.786,2.63,2.487,2.354,2.231,2.117,2.012,1.962,1.913,1.821,1.736,1.656,1.581,1.512,1.446,1.385,1.327,1.273,1.222,1.175,1.129,1.096,1.087,1.046,1.008,0.9721,0.9381,0.9057,0.8751,0.846,0.8184,0.7921,0.7673,0.7435,0.7209,0.6993,0.6789,0.6592,0.6406,0.6227,0.6057,0.5894,0.5738,0.5589,0.5446,0.5309,0.5178,0.5052,0.4931,0.4815,0.4704,0.4597,0.4494,0.4395,0.43,0.4208,0.2423])
        #mass_atten_coeff = np.interp(base_energy_array, energy_Mo, mass_atten_coeff_Mo)
        
        energy_Mo = adjust_duplicates(energy_Mo)
        interpolate_Mo = interp1d(energy_Mo, mass_atten_coeff_Mo, kind="linear", fill_value="extrapolate")
        mass_atten_coeff = interpolate_Mo(base_energy_array)

    elif filter_material_selection == "Rh (Z=45)":
        density = 12.4  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Rh", min_value=0.0, max_value=0.1, step=1E-6, value=0.0, key=f"filter_{filter_number}_thickness_Rh")
        energy_Rh = np.array([1,1.5,2,2.51,3,3.004,3.004,3.074,3.146,3.146,3.276,3.412,3.412,4,4.01,5,5.52,6,7.02,8,8.53,10,10.03,11.54,13.04,14.55,15,16.05,17.56,19.06,20,20.57,22.07,23.22,23.22,23.58,25.08,26.59,28.09,29.6,30,31.1,32.61,34.11,35.62,37.12,38.63,40,40.13,41.64,43.14,44.65,46.15,47.66,49.16,50,50.67,52.17,53.68,55.18,56.69,58.19,59.7,60,61.2,62.71,64.21,65.72,67.22,68.73,70.23,71.74,73.24,74.75,76.25,77.76,79.26,80,80.77,82.27,83.78,85.28,86.79,88.29,89.8,91.3,92.81,94.31,95.82,97.32,98.83,100,100.3,101.8,103.3,104.8,106.4,107.9,109.4,110.9,112.4,113.9,115.4,116.9,118.4,119.9,121.4,122.9,124.4,125.9,127.4,128.9,130.4,131.9,133.4,134.9,136.4,138,139.5,141,142.5,144,145.5,147,148.5,150,200])
        mass_atten_coeff_Rh = np.array([6170,2426,1214,690.7,444.2,442.7,1513,1422,1338,1847,1671,1512,1731,1170,1163,658.6,509.2,410.1,270.5,191.5,161.2,105.3,104.4,71.36,51.28,38.17,35.17,29.26,22.93,18.37,16.14,14.96,12.37,10.79,64.13,61.68,52.76,45.46,39.49,34.49,33.3,30.3,26.73,23.71,21.11,18.89,16.96,15.44,15.3,13.85,12.59,11.47,10.49,9.617,8.844,8.447,8.147,7.527,6.966,6.464,6.006,5.595,5.218,5.148,4.878,4.565,4.281,4.02,3.781,3.56,3.358,3.17,2.998,2.838,2.69,2.552,2.424,2.365,2.305,2.194,2.091,1.994,1.904,1.819,1.74,1.665,1.595,1.53,1.468,1.41,1.354,1.314,1.303,1.254,1.207,1.163,1.122,1.083,1.045,1.01,0.9763,0.9443,0.914,0.885,0.8576,0.8312,0.8063,0.7824,0.7597,0.7379,0.7172,0.6972,0.6783,0.66,0.6426,0.6259,0.6099,0.5945,0.5799,0.5657,0.5521,0.539,0.5265,0.5144,0.5029,0.4916,0.2742])
        #mass_atten_coeff = np.interp(base_energy_array, energy_Rh, mass_atten_coeff_Rh)
        
        energy_Rh = adjust_duplicates(energy_Rh)
        interpolate_Rh = interp1d(energy_Rh, mass_atten_coeff_Rh, kind="linear", fill_value="extrapolate")
        mass_atten_coeff = interpolate_Rh(base_energy_array)

    elif filter_material_selection == "Ag (Z=47)":
        density = 10.5  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Ag", min_value=0.0, max_value=0.1, step=1E-6, value=0.0, key=f"filter_{filter_number}_thickness_Ag")
        energy_Ag = np.array([1,1.5,2,2.51,3,3.351,3.351,3.436,3.524,3.524,3.662,3.806,3.806,4,4.01,5,5.52,6,7.02,8,8.53,10,10.03,11.54,13.04,14.55,15,16.05,17.56,19.06,20,20.57,22.07,23.58,25.08,25.51,25.51,26.59,28.09,29.6,30,31.1,32.61,34.11,35.62,37.12,38.63,40,40.13,41.64,43.14,44.65,46.15,47.66,49.16,50,50.67,52.17,53.68,55.18,56.69,58.19,59.7,60,61.2,62.71,64.21,65.72,67.22,68.73,70.23,71.74,73.24,74.75,76.25,77.76,79.26,80,80.77,82.27,83.78,85.28,86.79,88.29,89.8,91.3,92.81,94.31,95.82,97.32,98.83,100,100.3,101.8,103.3,104.8,106.4,107.9,109.4,110.9,112.4,113.9,115.4,116.9,118.4,119.9,121.4,122.9,124.4,125.9,127.4,128.9,130.4,131.9,133.4,134.9,136.4,138,139.5,141,142.5,144,145.5,147,148.5,150,200,])
        mass_atten_coeff_Ag = np.array([7037,2791,1401,797.9,513.6,388.7,1274,1198,1126,1547,1408,1282,1468,1305,1297,738.8,571.9,461.1,305,216.4,182.3,119.3,118.3,80.94,58.22,43.38,39.98,33.26,26.07,20.9,18.36,17.01,14.06,11.76,9.972,9.525,55.39,49.89,43.4,37.97,36.68,33.42,29.54,26.25,23.41,20.98,18.87,17.19,17.05,15.44,14.05,12.81,11.72,10.75,9.887,9.445,9.112,8.421,7.796,7.235,6.725,6.266,5.845,5.766,5.465,5.115,4.798,4.505,4.238,3.99,3.764,3.554,3.361,3.181,3.015,2.86,2.717,2.651,2.583,2.459,2.343,2.235,2.133,2.038,1.948,1.865,1.786,1.712,1.643,1.577,1.515,1.47,1.457,1.402,1.35,1.3,1.254,1.209,1.167,1.127,1.089,1.053,1.019,0.9865,0.9556,0.9259,0.8978,0.8708,0.8452,0.8206,0.7972,0.7747,0.7533,0.7327,0.7131,0.6942,0.6762,0.6588,0.6422,0.6262,0.6109,0.5961,0.582,0.5684,0.5553,0.5426,0.2972])
        #mass_atten_coeff = np.interp(base_energy_array, energy_Ag, mass_atten_coeff_Ag)
        
        energy_Ag = adjust_duplicates(energy_Ag)
        interpolate_Ag = interp1d(energy_Ag, mass_atten_coeff_Ag, kind="linear", fill_value="extrapolate")
        mass_atten_coeff = interpolate_Ag(base_energy_array)

    elif filter_material_selection == "Sn (Z=50)":
        density = 7.31  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Sn", min_value=0.0, max_value=1.0, step=0.005, value=0.0, key=f"filter_{filter_number}_thickness_Sn")
        energy_Sn = np.array([1,1.5,2,2.51,3,3.929,3.929,4,4.01,4.156,4.156,4.308,4.465,4.465,5,5.52,6,7.02,8,8.53,10,10.03,11.54,13.04,14.55,15,16.05,17.56,19.06,20,20.57,22.07,23.58,25.08,26.59,28.09,29.2,29.2,29.6,30,31.1,32.61,34.11,35.62,37.12,38.63,40,40.13,41.64,43.14,44.65,46.15,47.66,49.16,50,50.67,52.17,53.68,55.18,56.69,58.19,59.7,60,61.2,62.71,64.21,65.72,67.22,68.73,70.23,71.74,73.24,74.75,76.25,77.76,79.26,80,80.77,82.27,83.78,85.28,86.79,88.29,89.8,91.3,92.81,94.31,95.82,97.32,98.83,100,100.3,101.8,103.3,104.8,106.4,107.9,109.4,110.9,112.4,113.9,115.4,116.9,118.4,119.9,121.4,122.9,124.4,125.9,127.4,128.9,130.4,131.9,133.4,134.9,136.4,138,139.5,141,142.5,144,145.5,147,148.5,150,200])
        mass_atten_coeff_Sn = np.array([8155,3296,1665,951.6,614,311.4,925.7,939.4,933,846.9,1145,1055,971,1117,846.9,656.1,529.2,351.4,250.1,210.9,138.4,137.3,94.12,67.81,50.59,46.64,38.83,30.45,24.42,21.46,19.88,16.43,13.74,11.64,9.955,8.6,7.759,43.58,42.39,41.21,38.04,33.91,30.18,26.83,23.93,21.4,19.43,19.25,17.4,15.82,14.43,13.22,12.15,11.19,10.7,10.33,9.555,8.854,8.223,7.648,7.129,6.654,6.565,6.224,5.829,5.469,5.137,4.834,4.554,4.296,4.058,3.838,3.633,3.445,3.268,3.105,3.029,2.952,2.81,2.677,2.554,2.437,2.329,2.226,2.131,2.04,1.956,1.875,1.8,1.729,1.677,1.662,1.599,1.539,1.482,1.428,1.377,1.329,1.283,1.24,1.198,1.159,1.121,1.085,1.051,1.019,0.9877,0.9582,0.9298,0.9028,0.8769,0.8522,0.8284,0.8058,0.784,0.7632,0.7432,0.724,0.7056,0.6879,0.6709,0.6546,0.6388,0.6237,0.6091,0.326])
        #mass_atten_coeff = np.interp(base_energy_array, energy_Sn, mass_atten_coeff_Sn)
        
        energy_Sn = adjust_duplicates(energy_Sn)
        interpolate_Sn = interp1d(energy_Sn, mass_atten_coeff_Sn, kind="linear", fill_value="extrapolate")
        mass_atten_coeff = interpolate_Sn(base_energy_array)

    return mass_atten_coeff, filter_material_selection, density, filter_thickness

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

def calculate_effective_energy_and_hvl(energy_valid, energy_flux_normalised_filtered, filter_thickness):

    # Al details
    density = 2.7  # g/cm^3
    energy_Al = np.array([1.0,1.5,1.56,1.56,2,2.51,4.01,5.52,7.02,8.53,10.03,11.54,13.04,14.55,16.05,17.56,19.06,20.57,22.07,23.58,25.08,26.59,28.09,29.6,31.1,32.61,34.11,35.62,37.12,38.63,40.13,41.64,43.14,44.65,46.15,47.66,49.16,50.67,52.17,53.68,55.18,56.69,58.19,59.7,61.2,62.71,64.21,65.72,67.22,68.73,70.23,71.74,73.24,74.75,76.25,77.76,79.26,80.77,82.27,83.78,85.28,86.79,88.29,89.8,91.3,92.81,94.31,95.82,97.32,98.83,100.3,101.8,103.3,104.8,106.4,107.9,109.4,110.9,112.4,113.9,115.4,116.9,118.4,119.9,121.4,122.9,124.4,125.9,127.4,128.9,130.4,131.9,133.4,134.9,136.4,138,139.5,141,142.5,144,145.5,147,148.5,150,130.4,131.9,133.4,134.9,136.4,138,139.5,141,142.5,144,145.5,147,148.5,150,200])
    mass_atten_coeff_Al = np.array([1183,400.2,360,3955,2261,1271,356.7,145.1,72.64,41.07,25.43,16.75,11.63,8.372,6.239,4.767,3.734,2.979,2.423,1.999,1.675,1.419,1.218,1.056,0.9255,0.8178,0.7294,0.6552,0.5934,0.5408,0.4963,0.4579,0.4251,0.3965,0.3717,0.35,0.331,0.3141,0.2993,0.286,0.2742,0.2636,0.2542,0.2456,0.2378,0.2308,0.2244,0.2185,0.2132,0.2083,0.2038,0.1997,0.1958,0.1923,0.189,0.1859,0.1831,0.1804,0.1779,0.1755,0.1733,0.1713,0.1693,0.1675,0.1657,0.1641,0.1625,0.161,0.1596,0.1582,0.1569,0.1557,0.1545,0.1533,0.1522,0.1512,0.1502,0.1492,0.1483,0.1474,0.1465,0.1456,0.1448,0.144,0.1432,0.1425,0.1418,0.141,0.1404,0.1397,0.139,0.1384,0.1378,0.1372,0.1366,0.136,0.1354,0.1348,0.1343,0.1338,0.1332,0.1327,0.1322,0.1317,0.139,0.1384,0.1378,0.1372,0.1366,0.136,0.1354,0.1348,0.1343,0.1338,0.1332,0.1327,0.1322,0.1317,0.1188])

    # # Calculate the cumulative sum of the energy fluxes
    # cumulative_energy_flux = np.cumsum(energy_flux_normalised_filtered * np.diff(energy_valid, prepend=0))

    # # Normalise by the total AUC
    # normalised_cumulative_energy_flux = cumulative_energy_flux / np.trapz(energy_flux_normalised_filtered, energy_valid)
      
    # # Find the index for the where the AUC is 50%
    # indices = np.where(normalised_cumulative_energy_flux >= 0.5)[0]

    # Calculate the HVL mass attenuation coefficient for the Al filter
    mass_atten_coeff_eff = -np.log(energy_flux_normalised_filtered) / (filter_thickness * density / 10)

    # Interpolate to find the effective energy
    interpolate_energy = interp1d(mass_atten_coeff_Al, energy_Al, kind='linear', fill_value='extrapolate')
    energy_eff = interpolate_energy(mass_atten_coeff_eff)

    return energy_eff


# Set streamlit page to wide mode
st.set_page_config(layout="wide")

# Main function
if __name__ == "__main__":
    st.title("BremSpec")
    st.write("Bremsstrahlung X-ray Spectrum Visualiser") #`Decelerate and Illuminate` :P
    
    # Create two columns
    col1, col2, col3 = st.columns([1,2.5,0.5])

    # List of available plot styles
    plot_styles = ["classic","bmh","ggplot","Solarize_Light2","dark_background"]

    with col1: # elements in col1will display in the left column
        #st.subheader("Input Parameters")

        # User input for modality
        modality = st.selectbox("Modality", ["General X-ray", "Mammography", "Fluoroscopy (WIP)","CT (WIP)"])  # Add more modalities as needed

        # Set factors based on modality
        if modality == "General X-ray":
            tube_voltage_max = 150.0 # kV
            tube_voltage_min = 40.0
            tube_voltage_default = 125.0 
            tube_current_max = 500.0 # mA
            tube_current_min = 1.0
            tube_current_default = 500.0
            exposure_time_max = 1000.0 # ms
            exposure_time_min = 1.0
            exposure_time_default = 500.0
            current_time_product_max = 500.0 # mAs
            current_time_product_min = 0.0
            current_time_product_default = 250.0
            filters = ["Al (Z=13)", "Cu (Z=29)"]
            automatic_mode = "Automatic Exposure Control (AEC) (WIP)"

        elif modality == "Mammography":
            tube_voltage_max = 50.0
            tube_voltage_min = 10.0
            tube_voltage_default = 30.0
            tube_current_max = 100.0
            tube_current_min = 1.0
            tube_current_default = 50.0
            exposure_time_max = 200.0
            exposure_time_min = 1.0
            exposure_time_default = 100.0
            current_time_product_max = 100.0
            current_time_product_min = 1.0
            current_time_product_default = 20.0
            filters = ["Al (Z=13)","Mo (Z=42)", "Rh (Z=45)", "Ag (Z=47)"]
            automatic_mode = "Automatic Exposure Control (AEC) (WIP)"

        elif modality == "Fluoroscopy (WIP)":
            tube_voltage_max = 133.0
            tube_voltage_min = 40.0
            tube_voltage_default = 50.0
            tube_current_max = 500.0
            tube_current_min = 1.0
            tube_current_default = 100.0
            exposure_time_max = 1000.0
            exposure_time_min = 1.0
            exposure_time_default = 0.1
            pulse_width_max = 20.0 # ms
            pulse_width_min = 1.0
            pulse_width_default = 8.0
            filters = ["Al (Z=13)", "Cu (Z=29)"]
            automatic_mode = "Automatic Dose Rate Control (ADRC) (WIP)"

        elif modality == "CT (WIP)":
            tube_voltage_max = 140.0
            tube_voltage_min = 50.0
            tube_voltage_default = 120.0
            tube_current_max = 1000.0
            tube_current_min = 0.0
            tube_current_default = 500.0
            exposure_time_max = 2.0 # Rotation time
            exposure_time_min = 0.0
            exposure_time_default = 0.5
            filters = ["Al (Z=13)", "Cu (Z=29)", "Sn (Z=50)"]
            automatic_mode = "Automatic Exposure Control (AEC) (WIP)"

        # User input for mode
        mode = st.checkbox(automatic_mode)

        if "tube_voltage_old" not in st.session_state:
            st.session_state.tube_voltage_old = tube_voltage_default  # Default tube voltage

        if "current_time_product_old" not in st.session_state:
            st.session_state.current_time_product_old = tube_current_default*exposure_time_default/1000  # Default current-time product

        # User input for technique factors based on selected mode
        if mode: # Automatic mode
            
            tube_voltage = st.slider("Tube Voltage (kV)", min_value=int(tube_voltage_min), max_value=int(tube_voltage_max), value=int(tube_voltage_default))
            
            if modality == "CT":
                tube_current = 1/tube_voltage**2.0
                exposure_time = st.slider("Rotation Time (s)", min_value=exposure_time_min, max_value=exposure_time_max, value=exposure_time_default,format="%.2f")
            else:
                 # Calculate the new current-time product
                current_time_product = st.session_state.current_time_product_old*(st.session_state.tube_voltage_old/tube_voltage)**2.0
                current_time_product_display = st.write("Current-Time Product (mAs): ", round(current_time_product,0))
                
                # Update the old values for the next run
                st.session_state.tube_voltage_old = tube_voltage
                st.session_state.current_time_product_old = current_time_product

        else: # Manual mode
            tube_voltage = st.slider("Tube Voltage (kV)", min_value=int(tube_voltage_min), max_value=int(tube_voltage_max), value=int(tube_voltage_default))
            tube_current = st.slider("Tube Current (mA)", min_value=int(tube_current_min), max_value=int(tube_current_max), value=int(tube_current_default), step=1)
            if modality == "CT":
                exposure_time = st.slider("Rotation Time (ms)", min_value=exposure_time_min, max_value=exposure_time_max, value=exposure_time_default,format="%.0f")
            else:
                exposure_time = st.slider("Exposure Time (ms)", min_value=exposure_time_min, max_value=exposure_time_max, value=exposure_time_default,format="%.0f")
                current_time_product_display = st.write("Current-Time Product (mAs): ", round(tube_current*exposure_time / 1000,0))

        # Define a base energy array that all materials should conform to
        num_points = 1000 # higher number of points gives smoother plots but takes longer to compute
        energy_base_array = np.linspace(0, 150, num=num_points)  # Example: from 1 to 200 with 200 points

        # User input for filter materials
        mass_atten_coeff_1, filter_1_material, filter_1_density, filter_1_thickness = filter_selection_and_input(energy_base_array, 1, filters)

        # Determine a default value for the second filter that isn't the same as the first
        default_for_second_filter = filters[1] if filter_1_material == filters[0] else filters[0]
        mass_atten_coeff_2, filter_2_material, filter_2_density, filter_2_thickness = filter_selection_and_input(energy_base_array, 2, filters,default=default_for_second_filter)

        # Checkbox for showing charactersistic X-ray peaks
        show_characteristic_xray_peaks = st.checkbox("Show Characteristic X-ray Peaks", value=False)

        # Checkbox for showing the median beam energy
        show_median_energy = st.checkbox("Show Median Beam Energy", value=False)

        # Checkbox for showing the effective beam energy
        show_effective_energy = st.checkbox("Show Effective Beam Energy (WIP)", value=False)

        # User input for target material
        target_material = st.selectbox("Target Material", ["W (Z=74)", "Rh (Z=45)", "Mo (Z=42)"])
        if target_material == "W (Z=74)":
            Z = 74

            # Characteristic x-ray energies for tungsten (W) in keV
            # https://physics.nist.gov/PhysRefData/XrayTrans/Html/search.html
            # https://www.researchgate.net/publication/344795585_Simulation_of_X-Ray_Shielding_Effect_of_Different_Materials_Based_on_MCNP5#pf3
            # KL2, KL3, KM3, KN3, L2M2 (Select few)
            energy_char = np.array([57.98, 59.32, 67.25, 69.10, 8.97])

            # Estimated relative energy flux of characteristic x-ray peaks
            # These values are just crude estimates of the heights of the peaks relative to the maximum energy flux
            flux_peaks = np.array([1.1, 1.3, 0.8, 0.7, 0.1])

        elif target_material == "Rh (Z=45)":
            Z = 45
        elif target_material == "Mo (Z=42)":
            Z = 42


    with col3:
        # Dropdown for selecting plot style
        selected_style = st.selectbox("Select Plot Style", plot_styles)

        # Dropdown for selecting plotting colour
        selected_colour = st.selectbox("Select Plot Colour", ["royalblue","tomato","teal","lightgreen","lightsteelblue","cyan","magenta","gold","darkorange","darkviolet","crimson","deeppink","grey"])

        # Checkbox for turning grid on/off
        show_grid = st.checkbox("Show Grid", value=False)

        # Checkbox for scaling axes with selected kV
        scale_axes_with_kv = st.checkbox('Scale axes with selected kV')
        
        # Set the maximum tube voltage based selected kV or not for scaling the x-axis
        if scale_axes_with_kv:
            tube_voltage_max = tube_voltage
        
        y_axis_max = st.slider('Set maximum y-axis value:', min_value=0.0, max_value=1.0, value=1.0)

    with col2: # elements in col2 will be displayed in the right column

        # Calculate the spectrum and get energy values below the tube voltage
        if mode: # Automatic mode
            energy_valid, energy_flux_normalised = kramers_law(Z, energy_base_array, tube_voltage, tube_voltage, current_time_product=current_time_product,current_time_product_max=current_time_product_max)

        else: # Manual mode
            energy_valid, energy_flux_normalised = kramers_law(Z, energy_base_array, tube_voltage, tube_voltage, tube_current, tube_current_max, exposure_time, exposure_time_max)

        # Calculate the filtered spectrum
        energy_flux_normalised_filtered = energy_flux_normalised * relative_attenuation_mass_coeff(energy_base_array,filter_1_density, filter_1_thickness, mass_atten_coeff_1,tube_voltage) * relative_attenuation_mass_coeff(energy_base_array,filter_2_density, filter_2_thickness, mass_atten_coeff_2,tube_voltage)

        # Calculate the AUC percentage for the filtered spectrum
        auc_percentage = calculate_auc_percentage(energy_flux_normalised_filtered, energy_valid, 0, tube_voltage, tube_voltage_max)

        # Call calculate_median_energy_and_hvl
        median_energy_at_50pct_auc = calculate_median_energy(energy_valid, energy_flux_normalised_filtered)

        # Call calculate_effective_energy_and_hvl
        energy_eff = calculate_effective_energy_and_hvl(energy_valid, energy_flux_normalised_filtered, filter_1_thickness)
        
        ########## Visualise the spectrum ##########
        plt.style.use(selected_style)
        
        # Create a FontProperties object and set the font
        font = FontProperties()
        # font.set_family('Tahoma')

        fig, ax = plt.subplots(figsize=(14, 8),dpi=600)
   
        x_axis_limit = [0, tube_voltage_max] # Max energy is set by the tube voltage

        if show_characteristic_xray_peaks:
            # Add characteristic peaks to the spectrum
            energy_valid, energy_flux_normalised_filtered = add_characteristic_peaks(energy_valid, energy_flux_normalised_filtered, energy_char, flux_peaks, tube_voltage)

            # Plot the spectrum with characteristic peaks
            ax.plot(energy_valid, energy_flux_normalised_filtered,linestyle="-",linewidth=1.5,color=selected_colour)

            # Manually position each annotation
            annotations = [
                {"energy": energy_char[4], "peak": flux_peaks[4], "text": f"{energy_char[4]} keV", "xytext": (-20, 20)}, # L2M2
                {"energy": energy_char[1], "peak": flux_peaks[1], "text": f"{energy_char[1]} keV", "xytext": (20, 10)}, # KL3
                {"energy": energy_char[2], "peak": flux_peaks[2], "text": f"{energy_char[2]} keV", "xytext": (-20, 15)}, # KM3
                {"energy": energy_char[3], "peak": flux_peaks[3], "text": f"{energy_char[3]} keV", "xytext": (15, 0)},  # KN3
                {"energy": energy_char[0], "peak": flux_peaks[0], "text": f"{energy_char[0]} keV", "xytext": (-40, 10)}, # KL2
            ]

            # Annotate each peak
            for ann in annotations:
                if ann["energy"] <= tube_voltage: # Check if the peak energy is below the tube voltage
                    # Find the index where the peak is located in the energy array
                    peak_index = np.where(energy_valid == ann["energy"])[0]
                    if peak_index.size > 0:  # Check if the peak exists in the array
                        peak_index = peak_index[0]
                        y_value = energy_flux_normalised_filtered[peak_index]  # Correct y-value at the peak

                        if y_value < ann["peak"]:  # Only annotate if the peak is taller than the spectrum
                            # Create the annotation
                            ax.annotate(ann["text"],
                                        xy=(ann["energy"], y_value),
                                        xytext=ann["xytext"],
                                        textcoords="offset points",
                                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"),
                                        fontsize=8)
        else:
            # Plot the spectrum without characteristic peaks
            ax.plot(energy_valid, energy_flux_normalised_filtered,linestyle="-",linewidth=1.5,color=selected_colour)

        # Fill underneath the curve
        ax.fill_between(energy_valid, 0, energy_flux_normalised_filtered, color=selected_colour, alpha=0.4)

        if show_median_energy:
            # Add a vertical line for median energy at 50% AUC
            median_index = np.where(energy_valid >= median_energy_at_50pct_auc)[0][0]
            median_height = energy_flux_normalised_filtered[median_index]
            ax.plot([median_energy_at_50pct_auc, median_energy_at_50pct_auc], [0, median_height], color="navy", 
                    linestyle="--", linewidth=0.8, label=f"Median Energy: {median_energy_at_50pct_auc:.2f} keV")
        
            # Add annotation for the median energy
            ax.annotate(f"Median Energy: {median_energy_at_50pct_auc:.2f} keV", color="navy", 
                        xy=(median_energy_at_50pct_auc, median_height / 2),
                        xytext=(68, -20),  # Adjust these values to position your text
                        textcoords="offset points", 
                        ha="center",
                        fontsize=8,
                        fontproperties=font, 
                        #arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"),
            )
        
        if show_effective_energy:
            # Add a vertical line for effective energy
            ax.plot([energy_eff, energy_eff], [0, 1], color="darkgreen", linestyle="--", linewidth=0.8, label=f"Effective Energy: {energy_eff:.2f} keV")
        
            # Add annotation for the effective energy
            ax.annotate(f"Effective Energy: {energy_eff:.2f} keV", color="darkgreen", 
                        xy=(energy_eff, 0.5),
                        xytext=(68, -20),  # Adjust these values to position your text
                        textcoords="offset points", 
                        ha="center",
                        fontsize=8,
                        fontproperties=font, 
                        #arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"),
            )
        
        # # Annotate the AUC percentage on the plot
        ax.annotate(f"Relative AUC: {auc_percentage:.2f}% (of max factors, unfiltered)", 
                    color = "k",
                    xy=(0.64, 0.95), 
                    xycoords="axes fraction", 
                    fontsize=11,
                    fontproperties=font,
                    bbox=dict(boxstyle=None, fc="0.9"))

        ax.set_xlabel("Photon Energy E (keV)", fontsize=14)
        ax.set_ylabel("Relative Energy Flux", fontsize=14)
        ax.set_xlim(x_axis_limit)
        ax.set_ylim([0, y_axis_max])
        ax.set_xticks(np.arange(0, tube_voltage_max+1, 5))
        ax.set_yticks(np.arange(0, y_axis_max+0.05, 0.05))

        #ax.set_title(f"Bremsstrahlung Spectrum for Z={Z}")

        # Set grid based on checkbox
        ax.grid(show_grid)

        st.pyplot(fig)

        
