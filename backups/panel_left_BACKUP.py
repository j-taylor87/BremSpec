# panel_left.py

import streamlit as st
from utils.modality_settings import get_modality_settings

def render_panel_left(modalities: list[str]) -> dict:
    # User input for modality
    modality = st.selectbox("Modality", modalities)

    # Get settings for the selected modality
    settings = get_modality_settings(modality)

    # Extract settings for use
    tube_voltage_max = settings.get("tube_voltage_max", 0.0)
    tube_voltage_min = settings.get("tube_voltage_min", 0.0)
    tube_voltage_default = settings.get("tube_voltage_default", 0.0)

    tube_current_max = settings.get("tube_current_max", 0.0)
    tube_current_min = settings.get("tube_current_min", 0.0)
    tube_current_default = settings.get("tube_current_default", 0.0)

    # ---- Exposure / Pulse width ranges (ms) ----
    if modality == "Fluoroscopy":
        # Prefer the dedicated pulse_width_* keys; fall back to exposure_time_* if ever missing
        pulse_width_max = settings.get("pulse_width_max", settings.get("exposure_time_max", 100.0))
        pulse_width_min = settings.get("pulse_width_min", settings.get("exposure_time_min", 1.0))
        pulse_width_default = settings.get("pulse_width_default", settings.get("exposure_time_default", 10.0))

    else:
        exposure_time_max = settings.get("exposure_time_max", 0.0)
        exposure_time_min = settings.get("exposure_time_min", 0.0)
        exposure_time_default = settings.get("exposure_time_default", 0.0)

    current_time_product_max = settings.get("current_time_product_max", 0.0)
    current_time_product_min = settings.get("current_time_product_min", 0.0)
    current_time_product_default = settings.get("current_time_product_default", 0.0)

    filters = settings.get("filters", [])
    automatic_mode = settings.get("automatic_mode", "")

    mode = st.toggle(automatic_mode, value=False)

    if "tube_voltage_old" not in st.session_state:
        st.session_state.tube_voltage_old = tube_voltage_default  # Default tube voltage

    if "current_time_product_old" not in st.session_state:
        st.session_state.current_time_product_old = (
            tube_current_default * exposure_time_default / 1000.0
        )

    # Defaults so both branches have values to return
    tube_current = None
    exposure_time = None
    pulse_rate = None
    current_time_product = float(st.session_state.get("current_time_product_old", 0.0))

    # -------------------------------
    # Technique entry
    # -------------------------------
    font_family = "Arial, Helvetica, sans-serif"
    font_size = "17px"
    number_color = "#FF5733"

    if mode:  # Automatic mode
        tube_voltage = st.slider(
            "Tube Voltage (kV)",
            min_value=int(tube_voltage_min),
            max_value=int(tube_voltage_max),
            value=int(tube_voltage_default),
        )

        if modality == "CT":
            tube_current = 1 / tube_voltage**5.0  # your existing placeholder model
            exposure_time = st.slider(
                "Rotation Time (s)",
                min_value=exposure_time_min,
                max_value=exposure_time_max,
                value=exposure_time_default,
                format="%.2f",
            )
            # CT pitch (helical)
            ct_pitch = st.slider(
                "Pitch",
                min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                help="Table feed per rotation / total beam width"
            )

            # Keep auto scaling for downstream state
            current_time_product = (
                st.session_state.current_time_product_old
                * (st.session_state.tube_voltage_old / tube_voltage) ** 2.0
            )

            # Display CT metric: Effective mAs only (no extra mA display)
            effective_mas_ct = (float(tube_current) * float(exposure_time)) / float(ct_pitch)
            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Effective mAs (mA × rotation time / pitch): "
                f"<span style='color:{number_color};'><b>{effective_mas_ct:.1f}</b></span></p>",
                unsafe_allow_html=True,
            )

            # Update state
            st.session_state.tube_voltage_old = tube_voltage
            st.session_state.current_time_product_old = current_time_product

        elif modality == "Fluoroscopy":
            # Pulse width slider for duty-cycle display (does not change physics in auto)
            exposure_time = st.slider(
                "Pulse Width (ms)",
                min_value=pulse_width_min,
                max_value=pulse_width_max,
                value=pulse_width_default,
                format="%.0f",
            )
            pulse_rate = st.slider(
                "Pulse Rate (p/s)",
                min_value=0, max_value=30, value=30, step=1,
                help="Pulses per second"
            )

            # mAs per pulse (auto state scaling as before)
            current_time_product = (
                st.session_state.current_time_product_old
                * (st.session_state.tube_voltage_old / tube_voltage) ** 2.0
            )

            # Calculate duty cycle and effective mA
            duty_factor = float(pulse_rate) * (float(exposure_time) / 1000.0)  # fraction of a second
            # Effective mA (average over 1 s): mAs_per_pulse × pulses_per_second
            effective_ma = float(current_time_product) * float(pulse_rate)
            
            # Styled displays (match mAs style)
            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Duty cycle: "
                f"<span style='color:{number_color};'><b>{duty_factor:.1%}</b></span></p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Effective Tube Current (mA): "
                f"<span style='color:{number_color};'><b>{effective_ma:.2f}</b></span></p>",
                unsafe_allow_html=True,
            )

            st.session_state.tube_voltage_old = tube_voltage
            st.session_state.current_time_product_old = current_time_product

        elif modality == "Mammography":
            current_time_product = (
                st.session_state.current_time_product_old
                * (st.session_state.tube_voltage_old / tube_voltage) ** 2.0
            )
            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Current-Time Product (mAs): "
                f"<span style='color:{number_color};'><b>{round(current_time_product, 0)}</b></span></p>",
                unsafe_allow_html=True,
            )
            st.session_state.tube_voltage_old = tube_voltage
            st.session_state.current_time_product_old = current_time_product

        else:  # General X-ray (GXR)
            current_time_product = (
                st.session_state.current_time_product_old
                * (st.session_state.tube_voltage_old / tube_voltage) ** 2.0
            )
            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Current-Time Product (mAs): "
                f"<span style='color:{number_color};'><b>{round(current_time_product, 0)}</b></span></p>",
                unsafe_allow_html=True,
            )
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
            rotation_time_ms = st.slider(
                "Rotation Time (ms)",
                min_value=exposure_time_min,
                max_value=exposure_time_max,
                value=exposure_time_default,
                format="%.0f",
            )
            ct_pitch = st.slider(
                "Pitch",
                min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                help="Table feed per rotation / total beam width"
            )
            # downstream if needed; not displayed as generic mAs
            current_time_product = round((tube_current or 0) * (rotation_time_ms or 0) / 1000)

            # Display CT metric: Effective mAs only
            effective_mas_ct = (float(tube_current) * (float(rotation_time_ms) / 1000.0)) / float(ct_pitch)
            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Effective mAs (mA × rotation time / pitch): "
                f"<span style='color:{number_color};'><b>{effective_mas_ct:.1f}</b></span></p>",
                unsafe_allow_html=True,
            )

        elif modality == "Fluoroscopy":
            exposure_time = st.slider(
                "Pulse Width (ms)",
                min_value=pulse_width_min,
                max_value=pulse_width_max,
                value=pulse_width_default,
                format="%.0f",
            )
            pulse_rate = st.slider(
                "Pulse Rate (p/s)",
                min_value=0, max_value=30, value=30, step=1,
                help="Pulses per second"
            )

            # downstream if needed; not displayed as mAs for fluoro
            current_time_product = round((tube_current or 0) * (exposure_time or 0) / 1000)

            # Calculate duty cycle and effective mA
            duty_factor = float(pulse_rate) * (float(exposure_time) / 1000.0)  # fraction of a second
            # Effective mA (average over 1 s): mAs_per_pulse × pulses_per_second
            effective_ma = float(current_time_product) * float(pulse_rate)
            
            # Styled displays (match mAs style)
            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Duty cycle: "
                f"<span style='color:{number_color};'><b>{duty_factor:.1%}</b></span></p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Effective Tube Current (mA): "
                f"<span style='color:{number_color};'><b>{effective_ma:.2f}</b></span></p>",
                unsafe_allow_html=True,
            )

        else:  # Mammography / General X-ray
            exposure_time = st.slider(
                "Exposure Time (ms)",
                min_value=exposure_time_min,
                max_value=exposure_time_max,
                value=exposure_time_default,
                format="%.0f",
            )
            current_time_product = round((tube_current or 0) * (exposure_time or 0) / 1000)

            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Current-Time Product (mAs): "
                f"<span style='color:{number_color};'><b>{current_time_product}</b></span></p>",
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
            \n - The y-axis represents the relative energy flux (normalised). For Fluoroscopy mode, this is shown **per second** (includes pulse rate × pulse width).
            \n - Effective energy, HVL, and characteristic X-rays can be overlaid. Attenuation/transmission plots help visualise absorption edges.
            """
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
        "exposure_time": exposure_time,           # ms or s depending on CT auto; fine for our use
        "exposure_time_max": exposure_time_max,
        "current_time_product": float(current_time_product),
        "pulse_rate": int(pulse_rate) if pulse_rate is not None else None,  # NEW
        "show_characteristic_xray_peaks": bool(show_characteristic_xray_peaks),
        "show_effective_energy": bool(show_effective_energy),
        "show_median_energy": bool(show_median_energy),
        "show_mean_energy": bool(show_mean_energy),
        "show_peak_energy": bool(show_peak_energy),
    }
