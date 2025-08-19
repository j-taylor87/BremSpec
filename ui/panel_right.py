# panel_right.py

import streamlit as st
from utils.select_attenuation_and_load_data import select_attenuation

def render_panel_right(filters: list[str], data_dir: str, modality: str) -> dict:
    """
    Renders the right panel controls and returns all values needed downstream.
    Mirrors your original code structure and variable names.
    """

    # User input for target material
    target_material = st.selectbox("Anode Target Material", ["W (Z=74)", "Rh (Z=45)", "Mo (Z=42)"])

    # Default values for each filter
    default_filter_1 = filters[0]
    default_filter_2 = filters[1]
    default_filter_3 = filters[2]
    default_attenuation_4 = filters[2]  # kept for parity with original

    # Find the indices of the default values
    default_index_1 = filters.index(default_filter_1) if default_filter_1 in filters else 0
    default_index_2 = filters.index(default_filter_2) if default_filter_2 in filters else 0
    default_index_3 = filters.index(default_filter_3) if default_filter_3 in filters else 0

    # Colours for each material
    colour_material_1 = "red"
    colour_material_1a = "red"
    colour_material_2 = "green"
    colour_material_2a = "green"
    colour_material_3 = "darkviolet"
    colour_material_3a = "darkviolet"

    # Attenuator Materials
    with st.container():
        with st.popover("Attenuator Materials", use_container_width=True):
            filter_material_selection_1 = st.selectbox("Material 1", filters, index=default_index_1, key="filter_material_1")
            filter_material_selection_2 = st.selectbox("Material 2", filters, index=default_index_2, key="filter_material_2")
            filter_material_selection_3 = st.selectbox("Material 3", filters, index=default_index_3, key="filter_material_3")

    # Attenuation/Transmission Overlays
    with st.container():
        with st.popover("Attenuation/Transmission Overlays", use_container_width=True):
            st.markdown(f"<span style='color:{colour_material_1};'>Material 1</span>", unsafe_allow_html=True)
            show_attenuation_plot_filter_1 = st.checkbox("Attenuation 1")
            show_transmission_plot_filter_1 = st.checkbox("Transmission 1")

            st.markdown(f"<span style='color:{colour_material_2};'>Material 2</span>", unsafe_allow_html=True)
            show_attenuation_plot_filter_2 = st.checkbox("Attenuation 2")
            show_transmission_plot_filter_2 = st.checkbox("Transmission 2")

            st.markdown(f"<span style='color:{colour_material_3};'>Material 3</span>", unsafe_allow_html=True)
            show_attenuation_plot_filter_3 = st.checkbox("Attenuation 3")
            show_transmission_plot_filter_3 = st.checkbox("Transmission 3")

    # User input for filter materials (loads data)
    energy_base_array, mass_atten_coeff_1, filter_1_material, filter_1_density, filter_1_thickness = select_attenuation(
        1, filter_material_selection_1, data_dir, modality, colour_material_1
    )

    # Determine a default value for the second filter that isn't the same as the first (kept for parity)
    default_for_second_filter = filters[1] if filter_1_material == filters[0] else filters[0]
    energy_base_array_2, mass_atten_coeff_2, filter_2_material, filter_2_density, filter_2_thickness = select_attenuation(
        2, filter_material_selection_2, data_dir, modality, colour_material_2
    )

    # Determine a default value for the third filter that isn't the same as the first or second (kept for parity)
    default_for_third_filter = filters[2] if filter_1_material == filters[0] else filters[0]
    energy_base_array_3, mass_atten_coeff_3, filter_3_material, filter_3_density, filter_3_thickness = select_attenuation(
        3, filter_material_selection_3, data_dir, modality, colour_material_3
    )

    # Scatter Overlay
    with st.popover("Scatter Overlay", use_container_width=True):
        show_scatter_plot = st.toggle("Show Scatter Spectrum", value=False)
        scatter_angle_deg = st.slider("Compton Scatter Angle (degrees)", min_value=0, max_value=360, value=90, step=1)
        scatter_material = st.selectbox("Scatter Attenuator Material", filters, key="scatter_material")
        show_scatter_eff_hvl = st.checkbox("Show Effective Energy & HVL (scatter)", value=False)
        scatter_energy_base, scatter_mass_atten, scatter_material_name, scatter_density, scatter_thickness = select_attenuation(
            4, scatter_material, data_dir, modality, "deepskyblue"
        )
        scatter_y_scale = st.slider(
            "Scatter Intensity Scaling (% of primary maximum)",
            min_value=0.01,
            max_value=100.0,
            value=10.0,
            step=0.01,
            help="0.1% of primary typical for lateral scatter at 1 m",
        )

    # Return everything the rest of the app needs
    return {
        "target_material": target_material,
        # selections
        "filter_material_selection_1": filter_material_selection_1,
        "filter_material_selection_2": filter_material_selection_2,
        "filter_material_selection_3": filter_material_selection_3,
        # colors
        "colour_material_1": colour_material_1,
        "colour_material_1a": colour_material_1a,
        "colour_material_2": colour_material_2,
        "colour_material_2a": colour_material_2a,
        "colour_material_3": colour_material_3,
        "colour_material_3a": colour_material_3a,
        # overlays
        "show_attenuation_plot_filter_1": show_attenuation_plot_filter_1,
        "show_transmission_plot_filter_1": show_transmission_plot_filter_1,
        "show_attenuation_plot_filter_2": show_attenuation_plot_filter_2,
        "show_transmission_plot_filter_2": show_transmission_plot_filter_2,
        "show_attenuation_plot_filter_3": show_attenuation_plot_filter_3,
        "show_transmission_plot_filter_3": show_transmission_plot_filter_3,
        # loaded material data
        "energy_base_array": energy_base_array,
        "mass_atten_coeff_1": mass_atten_coeff_1,
        "filter_1_material": filter_1_material,
        "filter_1_density": filter_1_density,
        "filter_1_thickness": filter_1_thickness,
        "energy_base_array_2": energy_base_array_2,
        "mass_atten_coeff_2": mass_atten_coeff_2,
        "filter_2_material": filter_2_material,
        "filter_2_density": filter_2_density,
        "filter_2_thickness": filter_2_thickness,
        "energy_base_array_3": energy_base_array_3,
        "mass_atten_coeff_3": mass_atten_coeff_3,
        "filter_3_material": filter_3_material,
        "filter_3_density": filter_3_density,
        "filter_3_thickness": filter_3_thickness,
        # scatter controls + data
        "show_scatter_plot": show_scatter_plot,
        "scatter_angle_deg": scatter_angle_deg,
        "scatter_material": scatter_material,
        "show_scatter_eff_hvl": show_scatter_eff_hvl,
        "scatter_energy_base": scatter_energy_base,
        "scatter_mass_atten": scatter_mass_atten,
        "scatter_material_name": scatter_material_name,
        "scatter_density": scatter_density,
        "scatter_thickness": scatter_thickness,
        "scatter_y_scale": scatter_y_scale,
    }
