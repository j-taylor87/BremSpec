import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_data(data_dir, file_name):
    """
    Loads the interpolated mass attenuation coefficients from a CSV file.

    Parameters:
    data_dir (str): The directory where the data files are stored.
    file_name (str): The name of the data file to load.

    Returns:
    energy_base_array, mass_atten_coeff_cm2_g
    """
    interpolated_data_path = os.path.join(data_dir, file_name)
    df_mass_atten = pd.read_csv(interpolated_data_path)

    energy_base_array = df_mass_atten['energy_keV'].to_numpy()
    mass_atten_coeff_cm2_g = df_mass_atten['mass_atten_coeff_cm2_g'].to_numpy()
    
    return energy_base_array, mass_atten_coeff_cm2_g

def select_attenuation(filter_number, filter_material_selection, data_dir, modality, colour):
    """
    Selects a filter material and thickness, and reads its interpolated mass attenuation coefficients.

    Parameters:
    filter_number (int): The number of the filter (used for labeling in the user interface).
    filter_material_selection (str): The selected filter material.
    data_dir (str): The directory where the data files are stored.

    References: ICRU Report 44

    Returns:
    tuple:
      mass_atten_coeff (np.array): The mass attenuation coefficients corresponding to the base energy array.
      selected_filter (str): The selected filter material.
      filter_density (float): The density of the selected material (in g/cm^3).
      filter_thickness (float): The selected thickness of the filter (in mm).
    """
    if filter_material_selection == "Al (Z=13)":
        density = 2.7  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - Al</span>", unsafe_allow_html=True)
        if modality == "Mammography (WIP)":
            filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - Al", 
                                         min_value=0.0, 
                                         max_value=2.0, 
                                         step=0.0001, 
                                         value=0.0, 
                                         key=f"filter_{filter_number}_thickness_Al",
                                         label_visibility="collapsed")
        else:
            filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - Al", 
                                         min_value=0.0, 
                                         max_value=21.0, 
                                         step=0.05, 
                                         value=1.0, 
                                         key=f"filter_{filter_number}_thickness_Al",
                                         label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Al.csv"
        
    elif filter_material_selection == "Cu (Z=29)":
        density = 8.96  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - Cu</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - Cu", 
                                     min_value=0.0, 
                                     max_value=1.0, 
                                     step=0.01, 
                                     value=0.0, 
                                     key=f"filter_{filter_number}_thickness_Cu",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Cu.csv"

    elif filter_material_selection == "Be (Z=4)":
        density = 1.85  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - Be</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - Be", 
                                     min_value=0.0, 
                                     max_value=2.0, 
                                     step=0.01, 
                                     value=0.5, 
                                     key=f"filter_{filter_number}_thickness_Be",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Be.csv"

    elif filter_material_selection == "Ca (Z=20)":
        density = 1.55  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - Ca</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - Ca", 
                                     min_value=0.0, 
                                     max_value=1.0, 
                                     step=0.001, 
                                     value=0.0, 
                                     key=f"filter_{filter_number}_thickness_Ca",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Ca.csv"

    elif filter_material_selection == "CaSO4 (Gypsum) (Zeff~13)":
        density = 2.96  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - CaSO4</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - CaSO4", 
                                     min_value=0.0, 
                                     max_value=200.0, 
                                     step=1.0, 
                                     value=0.0, 
                                     key=f"filter_{filter_number}_thickness_CaSO4",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_CaSO4.csv"
        
    elif filter_material_selection == "Mo (Z=42)":
        density = 10.2  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - Mo</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - Mo", 
                                     min_value=0.0, 
                                     max_value=0.1, 
                                     step=1E-6, 
                                     value=0.0, 
                                     key=f"filter_{filter_number}_thickness_Mo",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Mo.csv"
        
    elif filter_material_selection == "Rh (Z=45)":
        density = 12.4  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - Rh</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - Rh", 
                                     min_value=0.0, 
                                     max_value=0.1, 
                                     step=1E-6, 
                                     value=0.0, 
                                     key=f"filter_{filter_number}_thickness_Rh",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Rh.csv"
        
    elif filter_material_selection == "Ag (Z=47)":
        density = 10.5  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - Ag</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - Ag", 
                                     min_value=0.0, 
                                     max_value=0.1, 
                                     step=1E-6, 
                                     value=0.0, 
                                     key=f"filter_{filter_number}_thickness_Ag",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Ag.csv"
    
    elif filter_material_selection == "I (Z=53)":
        density = 4.93  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - I</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - I", 
                                     min_value=0.0, 
                                     max_value=5.0, 
                                     step=0.001, 
                                     value=0.0, 
                                     key=f"filter_{filter_number}_thickness_I",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_I.csv"

    elif filter_material_selection == "Sn (Z=50)":
        density = 7.29  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - Sn</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - Sn", 
                                     min_value=0.0, 
                                     max_value=1.0, 
                                     step=0.001, 
                                     value=0.0, 
                                     key=f"filter_{filter_number}_thickness_Sn",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Sn.csv"

    elif filter_material_selection == "Pb (Z=82)":
        density = 11.34  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - Pb</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - Pb", 
                                     min_value=0.0, 
                                     max_value=20.0, 
                                     step=0.001, 
                                     value=0.0, 
                                     key=f"filter_{filter_number}_thickness_Pb",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_Pb.csv"

    elif filter_material_selection == "PMMA (Zeff~6.56)":
        density = 1.18  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - PMMA</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - PMMA", 
                                     min_value=0.0, 
                                     max_value=300.0, 
                                     step=1.0, 
                                     value=0.0, key=f"filter_{filter_number}_thickness_PMMA",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_PMMA.csv"
    
    elif filter_material_selection == "Soft Tissue (Zeff~7.52)":
        density = 1.03  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - Soft Tissue</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - Soft Tissue", 
                                     min_value=0.0, 
                                     max_value=300.0, 
                                     step=1.0, 
                                     value=0.0, 
                                     key=f"filter_{filter_number}_thickness_SoftTissue",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_TissueSoft4.csv"
    
    elif filter_material_selection == "Cortical Bone (Zeff~13.98)":
        density = 1.92  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - Cortical Bone</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - Cortical Bone", 
                                     min_value=0.0, 
                                     max_value=50.0, 
                                     step=0.1, 
                                     value=0.0, 
                                     key=f"filter_{filter_number}_thickness_CorticalBone",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_BoneCortical.csv"
    
    elif filter_material_selection == "Breast Tissue (Zeff~7.88)":
        density = 1.02  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - Breast Tissue</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number}  Thickness (mm) - Breast Tissue", 
                                     min_value=0.0, 
                                     max_value=80.0, 
                                     step=0.1, 
                                     value=0.0, 
                                     key=f"filter_{filter_number}_thickness_BreastTissue",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_TissueBreast.csv"
    
    elif filter_material_selection == "Adipose Tissue (Zeff~6.44)":
        density = 0.9  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - Adipose Tissue</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number}  Thickness (mm) - Adipose Tissue", 
                                     min_value=0.0, 
                                     max_value=100.0, 
                                     step=0.1, 
                                     value=0.0, 
                                     key=f"filter_{filter_number}_thickness_AdiposeTissue",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_TissueAdipose.csv"
    
    elif filter_material_selection == "Lung Tissue (Zeff~8.0)":
        density = 0.24  # g/cm^3
        st.markdown(f"<span style='color:{colour};'>Material {filter_number} Thickness (mm) - Lung Tissue</span>", unsafe_allow_html=True)
        filter_thickness = st.slider(f"Material {filter_number} Thickness (mm) - Lung Tissue", 
                                     min_value=0.0, 
                                     max_value=100.0, 
                                     step=0.1, 
                                     value=0.0, 
                                     key=f"filter_{filter_number}_thickness_LungTissue",
                                     label_visibility="collapsed")
        file_name = "interpolated_NIST_mass_attenuation_coeff_TissueLung.csv"

    else:
        st.warning("Select a valid filter material")
        return None, None, None, None

    # Load data using the cached function
    energy_base_array, mass_atten_coeff_cm2_g = load_data(data_dir, file_name)
    selected_filter = filter_material_selection
    # print(file_name)
    # print(mass_atten_coeff_cm2_g)

    return energy_base_array, mass_atten_coeff_cm2_g, selected_filter, density, filter_thickness