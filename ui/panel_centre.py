# panel_centre.py
import numpy as np
import streamlit as st
from scipy.interpolate import interp1d

from utils.calc_utils import (
    kramers_law,
    relative_attenuation_mass_coeff,
    calculate_auc_percentage,
    calculate_effective_energy_and_hvl,
    calculate_compton_scatter_spectrum,
    add_characteristic_peaks
)
from figures.figure_main import build_spectrum_figure
from figures.figure_secondary import (
    build_gxr_kap_3d,
    build_mammo_kedge_utilisation_figure,
    build_fluoro_pulse_waveform,
    build_ct_pitch_helix_figure,
)
from ui.ui_options_and_styles import PLOT_COLOURS

def render_panel_centre(left: dict, right: dict, plot_styles: list[str]) -> None:
    """
    left:  dict returned by render_panel_left(...)
    right: dict returned by render_panel_right(...)
    plot_styles: list of plotly templates to offer in the style selector
    """

    # ---- style state (read current) ----
    y_axis_max = st.session_state.get("y_axis_max", 1.0)
    scale_axes_with_kv = st.session_state.get("scale_axes_with_kv", False)  # kept for parity
    show_grid = st.session_state.get("show_grid", False)
    selected_style = st.session_state.get("selected_style", plot_styles[0])
    selected_colour = st.session_state.get("selected_colour", "royalblue")

    # ---- unpack inputs (left panel) ----
    mode = left["mode"]
    tube_voltage = left["tube_voltage"]
    tube_voltage_min = left["tube_voltage_min"]
    tube_voltage_max = left["tube_voltage_max"]
    tube_current = left["tube_current"]
    tube_current_max = left["tube_current_max"]
    exposure_time = left["exposure_time"]
    exposure_time_max = left["exposure_time_max"]
    current_time_product = left["current_time_product"]
    current_time_product_max = left["settings"].get("current_time_product_max", 0)

    show_characteristic_xray_peaks = left["show_characteristic_xray_peaks"]
    show_effective_energy = left["show_effective_energy"]
    show_median_energy = left["show_median_energy"]
    show_mean_energy = left["show_mean_energy"]
    show_peak_energy = left["show_peak_energy"]

    # ---- unpack inputs (right panel) ----
    target_material = right["target_material"]

    energy_base_array = right["energy_base_array"]
    mass_atten_coeff_1 = right["mass_atten_coeff_1"]
    filter_1_material = right["filter_1_material"]
    filter_1_density = right["filter_1_density"]
    filter_1_thickness = right["filter_1_thickness"]

    # Even though you load arrays for 2 & 3, we follow your original logic and
    # compute transmissions on the same base grid (energy_base_array).
    mass_atten_coeff_2 = right["mass_atten_coeff_2"]
    filter_2_density = right["filter_2_density"]
    filter_2_thickness = right["filter_2_thickness"]

    mass_atten_coeff_3 = right["mass_atten_coeff_3"]
    filter_3_density = right["filter_3_density"]
    filter_3_thickness = right["filter_3_thickness"]

    colour_material_1 = right["colour_material_1"]
    colour_material_1a = right["colour_material_1a"]
    colour_material_2 = right["colour_material_2"]
    colour_material_2a = right["colour_material_2a"]
    colour_material_3 = right["colour_material_3"]
    colour_material_3a = right["colour_material_3a"]

    show_transmission_plot_filter_1 = right["show_transmission_plot_filter_1"]
    show_transmission_plot_filter_2 = right["show_transmission_plot_filter_2"]
    show_transmission_plot_filter_3 = right["show_transmission_plot_filter_3"]
    show_attenuation_plot_filter_1 = right["show_attenuation_plot_filter_1"]
    show_attenuation_plot_filter_2 = right["show_attenuation_plot_filter_2"]
    show_attenuation_plot_filter_3 = right["show_attenuation_plot_filter_3"]

    show_scatter_plot = right["show_scatter_plot"]
    scatter_angle_deg = right["scatter_angle_deg"]
    scatter_material = right["scatter_material"]
    show_scatter_eff_hvl = right["show_scatter_eff_hvl"]
    scatter_energy_base = right["scatter_energy_base"]
    scatter_mass_atten = right["scatter_mass_atten"]
    scatter_density = right["scatter_density"]
    scatter_thickness = right["scatter_thickness"]
    scatter_y_scale = right["scatter_y_scale"]

    # ---- primary spectrum ----
    if mode:  # Automatic mode
        energy_valid, energy_flux_normalised = kramers_law(
            target_material,
            energy_base_array,
            tube_voltage,
            tube_voltage_max,
            tube_voltage_min,
            current_time_product=current_time_product,
            current_time_product_max=current_time_product_max,
        )
    else:  # Manual mode
        energy_valid, energy_flux_normalised = kramers_law(
            target_material,
            energy_base_array,
            tube_voltage,
            tube_voltage_max,
            tube_voltage_min,
            tube_current,
            tube_current_max,
            exposure_time,
            exposure_time_max,
        )

    # ---- filters (use base grid) ----
    mass_atten_coeff_1_valid, relative_attenuation_filter_1 = relative_attenuation_mass_coeff(
        energy_base_array, filter_1_density, filter_1_thickness, mass_atten_coeff_1, tube_voltage
    )
    mass_atten_coeff_2_valid, relative_attenuation_filter_2 = relative_attenuation_mass_coeff(
        energy_base_array, filter_2_density, filter_2_thickness, mass_atten_coeff_2, tube_voltage
    )
    mass_atten_coeff_3_valid, relative_attenuation_filter_3 = relative_attenuation_mass_coeff(
        energy_base_array, filter_3_density, filter_3_thickness, mass_atten_coeff_3, tube_voltage
    )

    #---- combine & AUC ----
    energy_flux_normalised_filtered = (
        energy_flux_normalised
        * relative_attenuation_filter_1
        * relative_attenuation_filter_2
        * relative_attenuation_filter_3
    )

    # --- Field area scaling (GXR/Mammo/Fluoro): normalise to MAX area ---
    if left["modality"] in ("General X-ray", "Mammography", "Fluoroscopy"):
        fa = float(left.get("field_area_cm2") or 0.0)
        fa_max = float(left["settings"].get("field_area_cm2_max", fa if fa > 0 else 1.0))
        field_scale = (fa / fa_max) if fa_max > 0 else 1.0
        energy_flux_normalised_filtered *= field_scale  # scales output & AUC to max-area = 1.0

    # --- Fluoro per-second scaling: per-pulse spectrum × (pulse_rate / 30) ---
    if left["modality"] == "Fluoroscopy":
        pulse_rate = float(left.get("pulse_rate") or 0.0)   # pulses per second
        ref_pulse_rate = 30.0                               # normalize to 30 p/s
        energy_flux_normalised_filtered *= (pulse_rate / ref_pulse_rate)

    auc_percentage = calculate_auc_percentage(
        energy_flux_normalised_filtered, energy_valid, 0, tube_voltage, tube_voltage_max
    )

    # --- Prepare arrays used for DISPLAY (peaks inserted) vs PHYSICS (original) ---
    display_energy = energy_valid
    display_flux   = energy_flux_normalised_filtered
    peak_annotations = []

    if show_characteristic_xray_peaks:
        # Insert peaks into a copy purely for plotting (does not change physics)
        display_energy, display_flux, peak_annotations = add_characteristic_peaks(
            target_material,
            energy_valid,
            energy_flux_normalised_filtered,
            tube_voltage,
            peak_gain=1.0,   # your ratios remain exactly as defined
        )

    # When scaling, treat the selected kV as the max for both AUC normalisation and x-axis
    x_axis_max = float(tube_voltage) if scale_axes_with_kv else float(tube_voltage_max)

    # ---- style state (read current) ----
    y_axis_max = st.session_state.get("y_axis_max", 1.0)
    scale_axes_with_kv = st.session_state.get("scale_axes_with_kv", False)
    show_grid = st.session_state.get("show_grid", False)
    selected_style = st.session_state.get("selected_style", plot_styles[0])
    selected_colour = st.session_state.get("selected_colour", "royalblue")

    # --- track previous axis-control values to know when to reset axes ---
    _prev = st.session_state.get("_axes_prev", {})
    y_axis_max_changed = ("y_axis_max" in _prev) and (_prev["y_axis_max"] != y_axis_max)
    xscale_changed     = ("scale_axes_with_kv" in _prev) and (_prev["scale_axes_with_kv"] != scale_axes_with_kv)

    # store current for next run
    st.session_state["_axes_prev"] = {
        "y_axis_max": y_axis_max,
        "scale_axes_with_kv": scale_axes_with_kv,
    }

    reset_axes = st.session_state.get("_first_main_render", True) or y_axis_max_changed or xscale_changed
        # tiny two-state toggler for uirevision *only* when we want to reset
    if reset_axes:
        st.session_state["_uirev_flip"] = not st.session_state.get("_uirev_flip", False)
    uirev_value = "main-spectrum-A" if st.session_state.get("_uirev_flip", False) else "main-spectrum-B"


    # ---- build figure (all visuals handled inside) ----
    fig = build_spectrum_figure(
        energy_valid=energy_valid,
        energy_flux_normalised_filtered=energy_flux_normalised_filtered,
        plot_energy_override=display_energy,
        plot_flux_override=display_flux,
        peak_annotations=peak_annotations,
        auc_percentage=auc_percentage,
        x_axis_max=x_axis_max,
        target_material=target_material,
        tube_voltage=tube_voltage,
        selected_colour=selected_colour,
        selected_style=selected_style,
        y_axis_max=y_axis_max,
        show_grid=show_grid,
        mass_atten_coeff_1_valid=mass_atten_coeff_1_valid,
        mass_atten_coeff_2_valid=mass_atten_coeff_2_valid,
        mass_atten_coeff_3_valid=mass_atten_coeff_3_valid,
        relative_attenuation_filter_1=relative_attenuation_filter_1,
        relative_attenuation_filter_2=relative_attenuation_filter_2,
        relative_attenuation_filter_3=relative_attenuation_filter_3,
        filter_1_density=filter_1_density,
        filter_1_material=filter_1_material,
        colour_material_1=colour_material_1, colour_material_1a=colour_material_1a,
        colour_material_2=colour_material_2, colour_material_2a=colour_material_2a,
        colour_material_3=colour_material_3, colour_material_3a=colour_material_3a,
        show_characteristic_xray_peaks=show_characteristic_xray_peaks,
        show_effective_energy=show_effective_energy,
        show_median_energy=show_median_energy,
        show_mean_energy=show_mean_energy,
        show_peak_energy=show_peak_energy,
        show_transmission_plot_filter_1=show_transmission_plot_filter_1,
        show_transmission_plot_filter_2=show_transmission_plot_filter_2,
        show_transmission_plot_filter_3=show_transmission_plot_filter_3,
        show_attenuation_plot_filter_1=show_attenuation_plot_filter_1,
        show_attenuation_plot_filter_2=show_attenuation_plot_filter_2,
        show_attenuation_plot_filter_3=show_attenuation_plot_filter_3,
        show_scatter_plot=show_scatter_plot,
        scatter_angle_deg=scatter_angle_deg,
        scatter_material=scatter_material,
        scatter_energy_base=scatter_energy_base,
        scatter_mass_atten=scatter_mass_atten,
        scatter_density=scatter_density,
        scatter_thickness=scatter_thickness,
        scatter_y_scale=scatter_y_scale,
        apply_axis_ranges=reset_axes,
        is_fluoro=(left["modality"] == "Fluoroscopy"),   # NEW
    )

    # ---- optional: scatter Eeff/HVL annotation (kept in center as in your original) ----
    if show_scatter_plot and show_scatter_eff_hvl:
        # Recreate the scaled scatter curve (same as figure.py) to compute HVL
        scatter_energy, scatter_flux = calculate_compton_scatter_spectrum(
            energy_valid,
            energy_flux_normalised_filtered,
            scatter_angle_deg,
            scatter_mass_atten,
            scatter_density,
            scatter_thickness,
            scatter_energy_base,
        )
        primary_max = np.max(energy_flux_normalised_filtered) if len(energy_flux_normalised_filtered) else 1.0
        scatter_flux_scaled = scatter_flux * (primary_max * (scatter_y_scale / 100.0))

        # Interpolate attenuation to scatter energy grid if lengths differ
        if len(scatter_mass_atten) != len(scatter_energy):
            mass_atten_interp = interp1d(
                scatter_energy_base, scatter_mass_atten, bounds_error=False, fill_value="extrapolate"
            )
            scatter_mass_atten_interp = mass_atten_interp(scatter_energy)
        else:
            scatter_mass_atten_interp = scatter_mass_atten

        eff_energy_scatter, hvl_scatter = calculate_effective_energy_and_hvl(
            scatter_energy, scatter_flux_scaled, scatter_mass_atten_interp, scatter_density
        )
        y_eff_sc = np.interp(eff_energy_scatter, scatter_energy, scatter_flux_scaled)

        fig.add_shape(
            type="line",
            x0=eff_energy_scatter, x1=eff_energy_scatter,
            y0=0, y1=y_eff_sc,
            line=dict(color="darkorange", width=1.8, dash="dot"),
        )
        fig.add_annotation(
            x=eff_energy_scatter,
            y=y_eff_sc,
            text=(f"E<sub>eff</sub> (scatter) = {eff_energy_scatter:.2f} keV<br>"
                    f"HVL = {hvl_scatter*10:.2f} mm {scatter_material}"),
            showarrow=False,
            xshift=0,
            yshift=30,
            font=dict(color="orange", size=16),
        )

    # ---- render plot ----
    fig.update_layout(uirevision="main-spectrum")  # constant; never tied to sliders
    st.plotly_chart(fig, use_container_width=True, key="fig-main")

    # mark first render done
    if st.session_state.get("_first_main_render", True):
        st.session_state._first_main_render = False

    # ---- controls below ----
    with st.container(border=True,):
        controls = st.columns([1.8, 1, 1, 1])
        with controls[0]:
            st.slider(
                "y-axis max",
                min_value=0.001, max_value=1.0, value=y_axis_max, step=0.001,
                # help="Set the maximum value of the Y axis",
                key="y_axis_max",
            )
        with controls[1]:
            st.checkbox(
                "Scale x-axis with selected kV",
                value=scale_axes_with_kv,
                key="scale_axes_with_kv",
            )
        with controls[2]:
            st.checkbox(
                "Show Grid",
                value=show_grid,
                key="show_grid",
            )
        with controls[3]:
            with st.popover("Plot Style"):
                st.selectbox(
                    "Select Plot Style",
                    plot_styles,
                    index=plot_styles.index(selected_style) if selected_style in plot_styles else 0,
                    key="selected_style",
                )
                st.selectbox(
                    "Select Plot Colour",
                    PLOT_COLOURS,
                    index=PLOT_COLOURS.index(selected_colour) if selected_colour in PLOT_COLOURS else 0,
                    key="selected_colour",
                )
    # --- optional secondary plot under the main figure ---
    st.toggle("Show Secondary Plot", value=st.session_state.get("show_secondary_plot", False), key="show_secondary_plot")

    if st.session_state.get("show_secondary_plot", False):

        modality = left["modality"]

        if modality == "General X-ray":
            sid_cm = float(left.get("sid_cm") or left["settings"].get("sid_cm_default", 110.0))
            area_cm2 = float(left.get("field_area_cm2") or left["settings"].get("field_area_cm2_default", 400.0))
            area_max_cm2 = float(left["settings"].get("field_area_cm2_max", area_cm2 if area_cm2>0.0 else 1.0))

            kap_fig = build_gxr_kap_3d(
                target_material=right["target_material"],
                energy_base_keV=right["energy_base_array"],
                kv_now=left["tube_voltage"],
                kv_min=left["tube_voltage_min"],
                kv_max=left["tube_voltage_max"],
                mas_now= left["current_time_product"],
                mas_max=left["tube_current_max"]*left["exposure_time_max"],
                mass_atten_1=right["mass_atten_coeff_1"], 
                dens_1=right["filter_1_density"], 
                thick_1_mm=right["filter_1_thickness"],
                mass_atten_2=right["mass_atten_coeff_2"], 
                dens_2=right["filter_2_density"], 
                thick_2_mm=right["filter_2_thickness"],
                mass_atten_3=right["mass_atten_coeff_3"], 
                dens_3=right["filter_3_density"], 
                thick_3_mm=right["filter_3_thickness"],
                field_area_cm2=area_cm2,
                field_area_max_cm2=area_max_cm2,
                sid_cm=sid_cm,
            )
            st.plotly_chart(kap_fig, use_container_width=True, key="gxr-kap-3d", config={"displayModeBar": True})

        elif modality == "Mammography":
            mam_fig = build_mammo_kedge_utilisation_figure(
                target_material=right["target_material"],
                filter_1_material=right["filter_1_material"],
                energy_base=right["energy_base_array"],
                kv_current=left["tube_voltage"],
                kv_min=left["tube_voltage_min"],
                kv_max=left["tube_voltage_max"],
                mass_atten_1=right["mass_atten_coeff_1"], dens_1=right["filter_1_density"], thick_1=right["filter_1_thickness"],
                mass_atten_2=right["mass_atten_coeff_2"], dens_2=right["filter_2_density"], thick_2=right["filter_2_thickness"],
                mass_atten_3=right["mass_atten_coeff_3"], dens_3=right["filter_3_density"], thick_3=right["filter_3_thickness"],
            )
            mam_fig.update_layout(uirevision="mammo-secondary-view")
            st.plotly_chart(mam_fig, use_container_width=True, key="fig-mammo-sec")          

        # --- pulsed fluoro square-wave under the main figure ---
        elif modality == "Fluoroscopy":
            pulse_rate_hz  = float(left.get("pulse_rate") or 30.0)
            # prefer explicit pulse_width if you now return it; else fall back to exposure_time (ms)
            pulse_width_ms = float(left.get("pulse_width") or left.get("exposure_time") or 10.0)
            peak_ma        = float(left.get("tube_current") or 0.0)
            y_max_ma       = float(left.get("tube_current_max") or peak_ma or 1.0)

            pulse_fig = build_fluoro_pulse_waveform(
                pulse_rate_hz=pulse_rate_hz,
                pulse_width_ms=pulse_width_ms,
                total_time_s=1.0,
                peak_ma=peak_ma,
                y_axis_max_ma=y_max_ma,  # <-- static 0..max mA
            )
            pulse_fig.update_layout(uirevision="fluoro-pulse-axes")  # keep constant
            st.plotly_chart(pulse_fig, use_container_width=True, config={"displayModeBar": False}, key="fig-pulse")

        elif modality == "CT":
            rot_time_s = float(left.get("exposure_time") or 0.0)
            if not left["mode"]:
                rot_time_s /= 1000.0

            pitch_val = float(left.get("ct_pitch") or 1.0)
            collimation_mm = float(left.get("ct_collimation_mm") or 80.0)

            ct_fig = build_ct_pitch_helix_figure(
                tube_current_mA=float(left.get("tube_current") or 0.0),
                rotation_time_s=rot_time_s,
                pitch=pitch_val,
                beam_width_mm=collimation_mm,  # <-- from slider
                n_turns=6,                     # keep the nice multi-turn view
                # radius_mm and axial_scale as you already set in the builder, or pass here if you expose them
            )
            ct_fig.update_layout(uirevision="ct-helix-view")         # keep constant
            st.plotly_chart(ct_fig, use_container_width=True, config={"displayModeBar": False}, key="fig-ct")