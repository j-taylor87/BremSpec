import streamlit as st
import pandas as pd
import os

def select_attenuation(filter_number, filters, data_dir, default=None):
    """
    Selects a filter material and thickness, and reads its interpolated mass attenuation coefficients.

    Parameters:
    filter_number (int): The number of the filter (used for labeling in the user interface).
    filters (list): A list of filter materials available for selection.
    data_dir (str): The directory where the data files are stored.
    default (str, optional): The default filter material to be preselected. Default is None.

    Returns:
    tuple:
      mass_atten_coeff (np.array): The mass attenuation coefficients corresponding to the base energy array.
      selected_filter (str): The selected filter material.
      filter_density (float): The density of the selected material (in g/cm^3).
      filter_thickness (float): The selected thickness of the filter (in mm).
    """
    # Use the default filter if provided, otherwise default to the first item in the list
    default_index = filters.index(default) if default else 0
    filter_material_selection = st.selectbox(f"Filter {filter_number} Material", filters, index=default_index, key=f"filter_material_{filter_number}")
  
    if filter_material_selection == "Al (Z=13)":
        density = 2.7  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Al", min_value=0.0, max_value=21.0, step=0.05, value=2.0, key=f"filter_{filter_number}_thickness_Al")
        interpolated_data_path = os.path.join(data_dir, "interpolated_NIST_mass_attenuation_coeff_Al.csv")
        
    elif filter_material_selection == "Cu (Z=29)":
        density = 8.96  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Cu", min_value=0.0, max_value=1.0, step=0.01, value=0.0, key=f"filter_{filter_number}_thickness_Cu")
        interpolated_data_path = os.path.join(data_dir, "interpolated_NIST_mass_attenuation_coeff_Cu.csv")
        
    elif filter_material_selection == "Mo (Z=42)":
        density = 10.2  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Mo", min_value=0.0, max_value=0.1, step=1E-6, value=0.0, key=f"filter_{filter_number}_thickness_Mo")
        interpolated_data_path = os.path.join(data_dir, "interpolated_NIST_mass_attenuation_coeff_Mo.csv")
        
    elif filter_material_selection == "Rh (Z=45)":
        density = 12.4  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Rh", min_value=0.0, max_value=0.1, step=1E-6, value=0.0, key=f"filter_{filter_number}_thickness_Rh")
        interpolated_data_path = os.path.join(data_dir, "interpolated_NIST_mass_attenuation_coeff_Rh.csv")
        
    elif filter_material_selection == "Ag (Z=47)":
        density = 10.5  # g/cm^3
        filter_thickness = st.slider(f"Filter {filter_number} Thickness (mm) - Ag", min_value=0.0, max_value=0.1, step=1E-6, value=0.0, key=f"filter_{filter_number}_thickness_Ag")
        interpolated_data_path = os.path.join(data_dir, "interpolated_NIST_mass_attenuation_coeff_Ag.csv")
    else:
        print("Select Filter")

    # Load interpolated data
    print("Loading data from:",interpolated_data_path)
    df_interpolated = pd.read_csv(interpolated_data_path)
    mass_atten_coeff_cm2_g = df_interpolated['mass_atten_coeff_cm2_g'].to_numpy()

    selected_filter = filter_material_selection

    return mass_atten_coeff_cm2_g, selected_filter, density, filter_thickness