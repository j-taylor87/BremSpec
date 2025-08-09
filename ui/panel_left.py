# panel_left.py

import streamlit as st
from utils.get_modality_settings import get_modality_settings


def render_panel_left(modalities: list[str]) -> dict:
    # User input for modality
    modality = st.selectbox("Modality", modalities)  # Add more modalities as needed

    # Get settings for the selected modality
    settings = get_modality_settings(modality)

    # Extract settings for use
    tube_voltage_max = settings.get("tube_voltage_max", 0)
    tube_voltage_min = settings.get("tube_voltage_min", 0)
    tube_voltage_default = settings.get("tube_voltage_default", 0)
    tube_current_max = settings.get("tube_current_max", 0)
    tube_current_min = settings.get("tube_current_min", 0)
    tube_current_default = settings.get("tube_current_default", 0)
    exposure_time_max = settings.get("exposure_time_max", 0)
    exposure_time_min = settings.get("exposure_time_min", 0)
    exposure_time_default = settings.get("exposure_time_default", 0)
    current_time_product_max = settings.get("current_time_product_max", 0)
    current_time_product_min = settings.get("current_time_product_min", 0)
    current_time_product_default = settings.get("current_time_product_default", 0)
    filters = settings.get("filters", [])
    automatic_mode = settings.get("automatic_mode", "")

    # User input for mode
    mode = st.toggle(automatic_mode, value=False)

    if "tube_voltage_old" not in st.session_state:
        st.session_state.tube_voltage_old = tube_voltage_default  # Default tube voltage

    if "current_time_product_old" not in st.session_state:
        st.session_state.current_time_product_old = tube_current_default * exposure_time_default / 1000.0  # Default

    # Defaults so both branches have values to return
    tube_current = None
    exposure_time = None
    current_time_product = float(st.session_state.get("current_time_product_old", 0.0))

    # User input for technique factors based on selected mode
    if mode:  # Automatic mode
        tube_voltage = st.slider(
            "Tube Voltage (kV)",
            min_value=int(tube_voltage_min),
            max_value=int(tube_voltage_max),
            value=int(tube_voltage_default),
        )

        if modality == "CT":
            tube_current = 1 / tube_voltage**5.0
            exposure_time = st.slider(
                "Rotation Time (s)",
                min_value=exposure_time_min,
                max_value=exposure_time_max,
                value=exposure_time_default,
                format="%.2f",
            )

            # Calculate the new current-time product
            current_time_product = st.session_state.current_time_product_old * (st.session_state.tube_voltage_old / tube_voltage) ** 2.0
            st.write("Current-Time Product (mAs): ", round(current_time_product, 0))

            # Update for next run
            st.session_state.tube_voltage_old = tube_voltage
            st.session_state.current_time_product_old = current_time_product

        elif modality == "Mammography":
            current_time_product = st.session_state.current_time_product_old * (st.session_state.tube_voltage_old / tube_voltage) ** 2.0
            st.write("Current-Time Product (mAs): ", round(current_time_product, 0))

            st.session_state.tube_voltage_old = tube_voltage
            st.session_state.current_time_product_old = current_time_product

        else:  # e.g. General X-ray
            current_time_product = st.session_state.current_time_product_old * (st.session_state.tube_voltage_old / tube_voltage) ** 2.0
            st.write("Current-Time Product (mAs): ", round(current_time_product, 0))

            st.session_state.tube_voltage_old = tube_voltage
            st.session_state.current_time_product_old = current_time_product

    else:  # Manual mode
        tube_voltage = st.slider(
            "Tube Voltage (kV)",
            min_value=int(tube_voltage_min),
            max_value=int(tube_voltage_max),
            value=int(tube_voltage_default),
        )
        tube_current = st.slider(
            "Tube Current (mA)",
            min_value=int(tube_current_min),
            max_value=int(tube_current_max),
            value=int(tube_current_default),
        )

        if modality == "CT":
            exposure_time = st.slider(
                "Rotation Time (ms)",
                min_value=exposure_time_min,
                max_value=exposure_time_max,
                value=exposure_time_default,
                format="%.0f",
            )
        elif modality == "Fluoroscopy":
            exposure_time = st.slider(
                "Pulse Width (ms)",
                min_value=exposure_time_min,
                max_value=exposure_time_max,
                value=exposure_time_default,
                format="%.0f",
            )
        else:
            exposure_time = st.slider(
                "Exposure Time (ms)",
                min_value=exposure_time_min,
                max_value=exposure_time_max,
                value=exposure_time_default,
                format="%.0f",
            )

            font_family = "Arial, Helvetica, sans-serif"
            font_size = "17px"
            number_color = "#FF5733"

            current_time_product = round(tube_current * exposure_time / 1000)

            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Current-Time Product (mAs): <span style='color:{number_color};'><b>{current_time_product}</b></span></p>",
                unsafe_allow_html=True,
            )

    # Toggles
    show_characteristic_xray_peaks = st.checkbox("Characteristic X-ray Peaks", value=False)
    show_effective_energy = st.checkbox("Effective Energy Eeff and HVL", value=False)
    show_median_energy = st.checkbox("Median Energy Eη", value=False)
    show_mean_energy = st.checkbox("Mean Energy Eμ", value=False)
    show_peak_energy = st.checkbox("Peak Energy Ep", value=False)

    with st.popover("Instructions", width="content"):
        st.markdown(
            """ - Select an X-ray imaging modality, technique factors (kV, mA, ms), and filter/target materials to see how these factors affect the 
            shape of the Bremsstrahlung X-ray spectrum.
            \n - The available energies along the x-axis will depend on the selected modality, and the x-axis can be set to scale with kV.
            \n - The y-axis represents the relative energy flux: the energy flux of each energy normalised to the maximum energy flux.
            \n - Scaling, zooming, and panning tools can be found on the top right of the graph when hovering over it with the mouse. "Axis Options" contains related settings.
            \n - The area under the curve (AUC), representing the normalised total beam energy across all available 
            energies, is displayed in the top-right corner of the plot. This is a relative percentage of the unfiltered beam at maximum technique factors of the selected modality, with a tungsten target. 
            If scaling x-axis with selected kV is selected, then the AUC calculation will take this as the maximum kV.
            \n - Effective beam energy can be displayed as a vertical line on the graph. HVL is also displayed, which is based on Material 1 attenuation.
            \n - Median, mean, and peak beam energy can also be displayed as vertical lines on the graph for comparison.
            \n - The characteristic X-rays of the anode target material can be displayed.
            \n - Attenuation and tranmission plots can be viewed for the different filter materials. This is especially useful for visualising absorption edges.
            \n - Attenuation and characteristic X-ray data obtained from NIST (2004) https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients"""
        )

    # Return everything the rest of the app needs
    return {
        "modality": modality,
        "settings": settings,
        "filters": filters,
        "mode": mode,
        "tube_voltage": int(tube_voltage),
        "tube_voltage_min": int(tube_voltage_min),
        "tube_voltage_max": int(tube_voltage_max),
        "tube_current": float(tube_current) if tube_current is not None else None,
        "tube_current_max": int(tube_current_max),
        "exposure_time": exposure_time,
        "exposure_time_max": exposure_time_max,
        "current_time_product": float(current_time_product),
        "show_characteristic_xray_peaks": bool(show_characteristic_xray_peaks),
        "show_effective_energy": bool(show_effective_energy),
        "show_median_energy": bool(show_median_energy),
        "show_mean_energy": bool(show_mean_energy),
        "show_peak_energy": bool(show_peak_energy),
    }
