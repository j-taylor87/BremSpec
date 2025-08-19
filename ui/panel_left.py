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

    field_area_min = float(settings.get("field_area_cm2_min", 100.0))
    field_area_max = float(settings.get("field_area_cm2_max", 1600.0))
    field_area_default = float(settings.get("field_area_cm2_default", 400.0))

    sid_min  = float(settings.get("sid_cm_min", 80.0))
    sid_max  = float(settings.get("sid_cm_max", 150.0))
    sid_def  = float(settings.get("sid_cm_default", 110.0))

    if modality == "Fluoroscopy":
        # Use dedicated pulse-width settings; fall back to exposure_time_* if ever missing
        pulse_width_min = settings.get("pulse_width_min", settings.get("exposure_time_min", 1.0))
        pulse_width_max = settings.get("pulse_width_max", settings.get("exposure_time_max", 100.0))
        pulse_width_default = settings.get("pulse_width_default", settings.get("exposure_time_default", 10.0))

        # Pulse rate defaul
        pulse_rate_default = settings.get("pulse_rate_default", 30)

        # Map onto exposure_time_* so downstream code continues to work
        exposure_time_min = pulse_width_min
        exposure_time_max = pulse_width_max
        exposure_time_default = pulse_width_default
    else:
        exposure_time_max = settings.get("exposure_time_max", 0.0)
        exposure_time_min = settings.get("exposure_time_min", 0.0)
        exposure_time_default = settings.get("exposure_time_default", 0.0)

    rotation_time_min = settings.get("rotation_time_min", settings.get("exposure_time_min", 100.0))
    rotation_time_max = settings.get("rotation_time_max", 2000.0)
    rotation_time_default = settings.get("rotation_time_default", settings.get("exposure_time_default", 500.0))

    scan_length_min = settings.get("scan_length_min_mm", 100.0)
    scan_length_max = settings.get("scan_length_max_mm", 2000.0)
    scan_length_default = settings.get("scan_length_default_mm", 500.0)

    filters = settings.get("filters", [])
    automatic_mode = settings.get("automatic_mode", "")

    # Mode
    mode = st.toggle(automatic_mode, value=False)

    # Initialise state anchors
    if "tube_voltage_old" not in st.session_state:
        st.session_state.tube_voltage_old = tube_voltage_default  # Default tube voltage

    if "current_time_product_old" not in st.session_state:
        # Use the modality-appropriate default width/time
        st.session_state.current_time_product_old = (
            float(tube_current_default) * float(exposure_time_default) / 1000.0
        )

    # Defaults so both branches have values to return
    tube_current = None
    exposure_time = None          # ms (except CT auto, where seconds)
    pulse_width = None            # explicit for Fluoro
    pulse_rate = None
    current_time_product = float(st.session_state.get("current_time_product_old", 0.0))
    field_area_cm2 = None
    sid_cm = None
    pulse_width_min = None
    pulse_width_max = None
    pulse_width_default = None
    ct_pitch = None
    ct_collimation_mm = None
    rotation_time_ms = None
    scan_length_mm = None

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

        if modality == "General X-ray":
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
            field_area_cm2 = st.slider(
                "Field size (cm²)",
                min_value=field_area_min, max_value=field_area_max, value=field_area_default, step=10.0,
                help="Collimated field area in cm².",
                key=f"field_area_cm2_{modality.replace(' ', '_').lower()}",
            )
            sid_cm = st.slider(
                "SID (cm)",
                min_value=sid_min, max_value=sid_max, value=sid_def, step=1.0,
                help="Source–Image Distance",
                key="sid_cm_gxr",
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
            field_area_cm2 = st.slider(
                "Field size (cm²)",
                min_value=field_area_min, max_value=field_area_max, value=field_area_default, step=10.0,
                help="Collimated field area in cm².",
                key=f"field_area_cm2_{modality.replace(' ', '_').lower()}",
            )
            st.session_state.tube_voltage_old = tube_voltage
            st.session_state.current_time_product_old = current_time_product

        elif modality == "Fluoroscopy":
            field_area_cm2 = st.slider(
                "Field size (cm²)",
                min_value=field_area_min, max_value=field_area_max, value=field_area_default, step=10.0,
                help="Collimated field area in cm².",
                key=f"field_area_cm2_{modality.replace(' ', '_').lower()}",
            )

            # Pulse width (ms) and pulse rate (pps)
            pulse_width = st.slider(
                "Pulse Width (ms)",
                min_value=float(exposure_time_min),
                max_value=float(exposure_time_max),
                value=float(exposure_time_default),
                format="%.0f",
            )
            exposure_time = pulse_width  # keep legacy key populated (ms)

            pulse_rate = st.slider(
                "Pulse Rate (p/s)",
                min_value=0, max_value=30, value=int(pulse_rate_default), step=1,
                help="Pulses per second"
            )

            duty_factor = float(pulse_rate) * (float(pulse_width) / 1000.0)
            effective_ma = float(current_time_product) * float(pulse_rate)

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

        elif modality == "CT":

            tube_current = 1 / tube_voltage**5.0  # placeholder model

            rotation_time_ms = st.slider(
                "Rotation Time (ms)",
                min_value=float(rotation_time_min),
                max_value=float(rotation_time_max),
                value=float(rotation_time_default),
                format="%.0f",
            )
            ct_pitch = st.slider(
                "Pitch",
                min_value=settings["pitch_min"], max_value=settings["pitch_max"],
                value=settings["pitch_default"], step=0.1, key="Pitch",
            )
            ct_collimation_mm = st.slider(
                "Collimation (mm)",
                min_value=settings["collimation_min_mm"], max_value=settings["collimation_max_mm"],
                value=settings["collimation_default_mm"], step=5.0, key="CT_Collimation_mm",
            )

            feed_per_rot = float(ct_pitch) * float(ct_collimation_mm)  # mm per rotation

            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Table feed / rotation: "
                f"<span style='color:{number_color};'><b>{feed_per_rot:.1f} mm</b></span> "
                f"(pitch × collimation)</p>",
                unsafe_allow_html=True,
            )
            current_time_product = (
                st.session_state.current_time_product_old
                * (st.session_state.tube_voltage_old / tube_voltage) ** 2.0
            )

            effective_mas_ct = (float(tube_current) * (float(rotation_time_ms) / 1000.0)) / float(ct_pitch)

            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Effective mAs (mA × rotation time / pitch): "
                f"<span style='color:{number_color};'><b>{effective_mas_ct:.1f}</b></span></p>",
                unsafe_allow_html=True,
            )

            scan_length_mm = st.slider(
                "Scan Length (mm)",
                min_value=float(scan_length_min),
                max_value=float(scan_length_max),
                value=float(scan_length_default),
                step=1.0,
                key="scan_length_mm",
            )

            st.session_state.tube_voltage_old = tube_voltage
            st.session_state.current_time_product_old = current_time_product

        else:
            st.error("Unsupported modality for automatic mode.")
            return {}

    else:  # Manual mode
        tube_voltage = st.slider(
            "Tube Voltage (kV)",
            min_value=int(tube_voltage_min),
            max_value=int(tube_voltage_max),
            value=int(tube_voltage_default),
        )

        if modality == "General X-ray":
            tube_current = st.slider(
                "Tube Current (mA)",
                min_value=int(tube_current_min),
                max_value=int(tube_current_max),
                value=int(tube_current_default),
            )
            exposure_time = st.slider(
                "Exposure Time (ms)",
                min_value=float(exposure_time_min),
                max_value=float(exposure_time_max),
                value=float(exposure_time_default),
                format="%.0f",
            )

            current_time_product = round((tube_current or 0) * (exposure_time or 0) / 1000)
            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Current-Time Product (mAs): "
                f"<span style='color:{number_color};'><b>{round(current_time_product, 0)}</b></span></p>",
                unsafe_allow_html=True,
            )
                                         
            field_area_cm2 = st.slider(
                "Field size (cm²)",
                min_value=field_area_min, max_value=field_area_max, value=field_area_default, step=10.0,
                help="Collimated field area in cm².",
                key=f"field_area_cm2_{modality.replace(' ', '_').lower()}",
            )
            sid_cm = st.slider(
                "SID (cm)",
                min_value=sid_min, max_value=sid_max, value=sid_def, step=1.0,
                help="Source–Image Distance",
                key="sid_cm_gxr",
            )

        elif modality == "Mammography":
            tube_current = st.slider(
                "Tube Current (mA)",
                min_value=int(tube_current_min),
                max_value=int(tube_current_max),
                value=int(tube_current_default),
            )
            exposure_time = st.slider(
                "Exposure Time (ms)",
                min_value=float(exposure_time_min),
                max_value=float(exposure_time_max),
                value=float(exposure_time_default),
                format="%.0f",
            )
            current_time_product = round((tube_current or 0) * (exposure_time or 0) / 1000)
            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Current-Time Product (mAs): "
                f"<span style='color:{number_color};'><b>{round(current_time_product, 0)}</b></span></p>",
                unsafe_allow_html=True,
            )
            field_area_cm2 = st.slider(
                "Field size (cm²)",
                min_value=field_area_min, max_value=field_area_max, value=field_area_default, step=10.0,
                help="Collimated field area in cm².",
                key=f"field_area_cm2_{modality.replace(' ', '_').lower()}",
            )

        elif modality == "Fluoroscopy":
            tube_current = st.slider(
                "Peak Tube Current (mA)",
                min_value=int(tube_current_min),
                max_value=int(tube_current_max),
                value=int(tube_current_default),
            )

            field_area_cm2 = st.slider(
                "Field size (cm²)",
                min_value=field_area_min, max_value=field_area_max, value=field_area_default, step=10.0,
                help="Collimated field area in cm².",
                key=f"field_area_cm2_{modality.replace(' ', '_').lower()}",
            )
            pulse_width = st.slider(
                "Pulse Width (ms)",
                min_value=float(exposure_time_min),
                max_value=float(exposure_time_max),
                value=float(exposure_time_default),
                format="%.0f",
            )
            exposure_time = pulse_width  # keep legacy key populated (ms)

            pulse_rate = st.slider(
                "Pulse Rate (p/s)",
                min_value=0, max_value=30, value=int(pulse_rate_default), step=1,
                help="Pulses per second"
            )

            current_time_product = round((tube_current or 0) * (pulse_width or 0) / 1000)

            duty_factor = float(pulse_rate) * (float(pulse_width) / 1000.0)
            effective_ma = float(current_time_product) * float(pulse_rate)

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
        
        elif modality == "CT":
            tube_current = st.slider(
                "Tube Current (mA)",
                min_value=int(tube_current_min),
                max_value=int(tube_current_max),
                value=int(tube_current_default),
            )
            rotation_time_ms = st.slider(
                "Rotation Time (ms)",
                min_value=float(rotation_time_min),
                max_value=float(rotation_time_max),
                value=float(rotation_time_default),
                format="%.0f",
            )
            ct_pitch = st.slider(
                "Pitch",
                min_value=settings["pitch_min"], max_value=settings["pitch_max"],
                value=settings["pitch_default"], step=0.1, key="Pitch",
            )

            ct_collimation_mm = st.slider(
                "Collimation (mm)",
                min_value=settings["collimation_min_mm"], max_value=settings["collimation_max_mm"],
                value=settings["collimation_default_mm"], step=5.0, key="CT_Collimation_mm",
            )

            feed_per_rot = float(ct_pitch) * float(ct_collimation_mm)

            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Table feed / rotation: "
                f"<span style='color:{number_color};'><b>{feed_per_rot:.1f} mm</b></span> "
                f"(pitch × collimation)</p>",
                unsafe_allow_html=True,
            )

            current_time_product = round((tube_current or 0) * (rotation_time_ms or 0) / 1000)

            effective_mas_ct = (float(tube_current) * (float(rotation_time_ms) / 1000.0)) / float(ct_pitch)

            st.markdown(
                f"<p style='font-family:{font_family}; font-size:{font_size};'>"
                f"Effective mAs (mA × rotation time / pitch): "
                f"<span style='color:{number_color};'><b>{effective_mas_ct:.1f}</b></span></p>",
                unsafe_allow_html=True,
            )

            scan_length_mm = st.slider(
                "Scan Length (mm)",
                min_value=float(scan_length_min),
                max_value=float(scan_length_max),
                value=float(scan_length_default),
                step=1.0,
                key="scan_length_mm",
            )

        else:
            st.error("Unsupported modality for manual mode.")
            return {}
        
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
        "exposure_time": exposure_time,              
        "exposure_time_max": exposure_time_max,
        "current_time_product": float(current_time_product),

        "field_area_cm2": float(field_area_cm2) if field_area_cm2 is not None else field_area_default,
        "sid_cm": float(sid_cm) if sid_cm is not None else None,

        "pulse_width": float(pulse_width) if pulse_width is not None else None,
        "pulse_width_min": float(pulse_width_min) if pulse_width_min is not None else None,
        "pulse_width_max": float(pulse_width_max) if pulse_width_max is not None else None,
        "pulse_width_default": float(pulse_width_default) if pulse_width_default is not None else None,
        "pulse_rate": int(pulse_rate) if pulse_rate is not None else None,

        "rotation_time_ms": float(rotation_time_ms) if rotation_time_ms is not None else None,
        "ct_pitch": float(ct_pitch) if ct_pitch is not None else None,
        "ct_collimation_mm": float(ct_collimation_mm) if ct_collimation_mm is not None else None,
        "scan_length_mm": float(scan_length_mm) if scan_length_mm is not None else None,

        "show_characteristic_xray_peaks": bool(show_characteristic_xray_peaks),
        "show_effective_energy": bool(show_effective_energy),
        "show_median_energy": bool(show_median_energy),
        "show_mean_energy": bool(show_mean_energy),
        "show_peak_energy": bool(show_peak_energy),
    }
