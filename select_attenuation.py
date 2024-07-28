import streamlit as st
import pandas as pd
import os

# @st.cache_data(experimental_allow_widgets=True)
@st.cache_data

def load_data(data_dir, file_name):
    """
    Loads the interpolated mass attenuation coefficients from a CSV file.

    Parameters:
    data_dir (str): The directory where the data files are stored.
    file_name (str): The name of the data file to load.

    Returns:
    np.array: The mass attenuation coefficients.
    """
    interpolated_data_path = os.path.join(data_dir, file_name)
    df_interpolated = pd.read_csv(interpolated_data_path)
    mass_atten_coeff_cm2_g = df_interpolated['mass_atten_coeff_cm2_g'].to_numpy()
    return mass_atten_coeff_cm2_g

def select_attenuation(filter_number, filter_material_selection, data_dir):
    """
    Selects a filter material and thickness, and reads its interpolated mass attenuation coefficients.

    Parameters:
    filter_number (int): The number of the filter (used for labeling in the user interface).
    filter_material_selection (str): The selected filter material.
    data_dir (str): The directory where the data files are stored.

    Returns:
    tuple:
      mass_atten_coeff (np.array): The mass attenuation coefficients corresponding to the base energy array.
      selected_filter (str): The selected filter material.
      filter_density (float): The density of the selected material (in g/cm^3).
      filter_thickness (float): The selected thickness of the filter (in mm).
    """
    if filter_material_selection == "Al (Z=13)":
        density = 2.7  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Al", min_value=0.0, max_value=21.0, step=0.05, value=2.0, key=f"filter_{filter_number}_thickness_Al")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Al.csv"
        
    elif filter_material_selection == "Cu (Z=29)":
        density = 8.96  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Cu", min_value=0.0, max_value=1.0, step=0.01, value=0.0, key=f"filter_{filter_number}_thickness_Cu")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Cu.csv"

    elif filter_material_selection == "Be (Z=4)":
        density = 1.85  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Be", min_value=0.0, max_value=2.0, step=0.01, value=0.5, key=f"filter_{filter_number}_thickness_Be")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Be.csv"
        
    elif filter_material_selection == "Mo (Z=42)":
        density = 10.2  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Mo", min_value=0.0, max_value=0.1, step=1E-6, value=0.0, key=f"filter_{filter_number}_thickness_Mo")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Mo.csv"
        
    elif filter_material_selection == "Rh (Z=45)":
        density = 12.4  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Rh", min_value=0.0, max_value=0.1, step=1E-6, value=0.0, key=f"filter_{filter_number}_thickness_Rh")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Rh.csv"
        
    elif filter_material_selection == "Ag (Z=47)":
        density = 10.5  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Ag", min_value=0.0, max_value=0.1, step=1E-6, value=0.0, key=f"filter_{filter_number}_thickness_Ag")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Ag.csv"

    elif filter_material_selection == "PMMA (Zeff~6.56)":
        density = 1.18  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number}/Attenuator  Thickness (mm) - PMMA", min_value=0.0, max_value=300.0, step=5.0, value=0.0, key=f"filter_{filter_number}_thickness_PMMA")
        file_name = "interpolated_NIST_mass_attenuation_coeff_PMMA.csv"
    
    elif filter_material_selection == "Soft Tissue (Zeff~7.52)":
        density = 1.03  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number}/Attenuator Thickness (mm) - Soft Tissue", min_value=0.0, max_value=300.0, step=5.0, value=0.0, key=f"filter_{filter_number}_thickness_SoftTissue")
        file_name = "interpolated_NIST_mass_attenuation_coeff_TissueSoft4.csv"
    
    elif filter_material_selection == "Cortical Bone (Zeff~13.98)":
        density = 1.92  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number}/Attenuator Thickness (mm) - Cortical Bone", min_value=0.0, max_value=100.0, step=5.0, value=0.0, key=f"filter_{filter_number}_thickness_CorticalBone")
        file_name = "interpolated_NIST_mass_attenuation_coeff_BoneCortical.csv"
    
    elif filter_material_selection == "Breast Tissue (Zeff~7.88)":
        density = 1.02  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number}/Attenuator  Thickness (mm) - Breast Tissue", min_value=0.0, max_value=150.0, step=5.0, value=0.0, key=f"filter_{filter_number}_thickness_BreastTissue")
        file_name = "interpolated_NIST_mass_attenuation_coeff_TissueBreast.csv"

    
    else:
        st.warning("Select a valid filter material")
        return None, None, None, None

    # Load data using the cached function
    mass_atten_coeff_cm2_g = load_data(data_dir, file_name)
    selected_filter = filter_material_selection

    return mass_atten_coeff_cm2_g, selected_filter, density, filter_thickness