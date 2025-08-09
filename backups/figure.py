# figure.py

import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

from utils.calc_utils import (
    add_characteristic_peaks,
    calculate_effective_energy_and_hvl,
    calculate_median_energy,
    calculate_mean_energy,
    calculate_peak_energy,
    calculate_compton_scatter_spectrum,
)

def build_spectrum_figure(
    *,
    # core data
    energy_valid,
    energy_flux_normalised_filtered,
    auc_percentage,
    tube_voltage_max,
    target_material,
    tube_voltage,
    # style
    selected_colour,
    selected_style,
    y_axis_max,
    show_grid,
    # overlays data
    mass_atten_coeff_1_valid,
    mass_atten_coeff_2_valid,
    mass_atten_coeff_3_valid,
    relative_attenuation_filter_1,
    relative_attenuation_filter_2,
    relative_attenuation_filter_3,
    filter_1_density,
    filter_1_material,
    colour_material_1, colour_material_1a,
    colour_material_2, colour_material_2a,
    colour_material_3, colour_material_3a,
    # toggles
    show_characteristic_xray_peaks,
    show_effective_energy,
    show_median_energy,
    show_mean_energy,
    show_peak_energy,
    show_transmission_plot_filter_1,
    show_transmission_plot_filter_2,
    show_transmission_plot_filter_3,
    show_attenuation_plot_filter_1,
    show_attenuation_plot_filter_2,
    show_attenuation_plot_filter_3,
    # scatter
    show_scatter_plot,
    scatter_angle_deg,
    scatter_material,
    scatter_energy_base,
    scatter_mass_atten,
    scatter_density,
    scatter_thickness,
    scatter_y_scale,
):
    fig = go.Figure()

    # base spectrum
    fig.add_trace(go.Scatter(
        x=energy_valid,
        y=energy_flux_normalised_filtered,
        mode='lines',
        line=dict(color=selected_colour, width=1.5),
        name="Spectrum"
    ))

    # characteristic peaks
    if show_characteristic_xray_peaks:
        e_with_peaks, y_with_peaks, annotations = add_characteristic_peaks(
            target_material, energy_valid, energy_flux_normalised_filtered, tube_voltage
        )
        for ann in annotations:
            if ann["energy"] <= tube_voltage:
                idx = np.where(e_with_peaks == ann["energy"])[0]
                if idx.size > 0:
                    y_val = y_with_peaks[int(idx[0])]
                    if y_val < ann["peak"]:
                        fig.add_annotation(
                            x=ann["energy"],
                            y=y_val * 0.95,
                            text=ann["text"],
                            showarrow=True,
                            arrowhead=3,
                            arrowsize=1,
                            arrowwidth=1.2,
                            ax=ann["xytext"][0],
                            ay=ann["xytext"][1],
                            font=dict(size=16),
                        )

    # fill under curve
    r, g, b = [int(255 * c) for c in mcolors.to_rgb(selected_colour)]
    fill_color_rgba = f"rgba({r},{g},{b},0.2)"
    fig.add_trace(go.Scatter(
        x=energy_valid,
        y=energy_flux_normalised_filtered,
        mode='lines',
        fill='tozeroy',
        line=dict(color=selected_colour, width=1.5),
        fillcolor=fill_color_rgba,
        name="Filled Area"
    ))

    # effective energy + HVL (material 1)
    if show_effective_energy:
        eff_e, t_hvl = calculate_effective_energy_and_hvl(
            energy_valid, energy_flux_normalised_filtered, mass_atten_coeff_1_valid, filter_1_density
        )
        eff_idx = np.where(energy_valid >= eff_e)[0][0]
        eff_y = energy_flux_normalised_filtered[eff_idx]
        fig.add_shape(type="line", x0=eff_e, x1=eff_e, y0=0, y1=eff_y, line=dict(color="blue", width=2, dash="dash"))
        fig.add_annotation(
            x=eff_e, y=eff_y * 1.25,
            text=(f"E<sub>eff</sub> = {eff_e:.2f} keV<br>HVL = {t_hvl*10:.2f} mm {filter_1_material}"),
            showarrow=False, font=dict(color="blue", size=18)
        )

    # median / mean / peak lines
    if show_median_energy:
        med_e = calculate_median_energy(energy_valid, energy_flux_normalised_filtered)
        med_idx = np.where(energy_valid >= med_e)[0][0]
        med_y = energy_flux_normalised_filtered[med_idx]
        fig.add_shape(type="line", x0=med_e, x1=med_e, y0=0, y1=med_y, line=dict(color="green", width=2, dash="dash"))
        fig.add_annotation(x=med_e, y=med_y / 2, text=f"E<sub>η</sub> = {med_e:.2f} keV", showarrow=False, font=dict(color="green", size=16))

    if show_mean_energy:
        mean_e = calculate_mean_energy(energy_valid, energy_flux_normalised_filtered)
        mean_idx = np.where(energy_valid >= mean_e)[0][0]
        mean_y = energy_flux_normalised_filtered[mean_idx]
        fig.add_shape(type="line", x0=mean_e, x1=mean_e, y0=0, y1=mean_y, line=dict(color="blueviolet", width=3, dash="dot"))
        fig.add_annotation(x=mean_e, y=mean_y / 8, text=f"E<sub>μ</sub> = {mean_e:.2f} keV", showarrow=False, font=dict(color="blueviolet", size=16))

    if show_peak_energy:
        peak_e = calculate_peak_energy(energy_valid, energy_flux_normalised_filtered)
        peak_y = energy_flux_normalised_filtered[int(np.argmax(energy_flux_normalised_filtered))]
        fig.add_shape(type="line", x0=peak_e, x1=peak_e, y0=0, y1=peak_y, line=dict(color="darkorange", width=3, dash="dashdot"))
        fig.add_annotation(x=peak_e, y=peak_y * 1.05, text=f"E<sub>p</sub> = {peak_e:.2f} keV", showarrow=False, font=dict(color="orange", size=16))

    # transmission / attenuation overlays
    if show_transmission_plot_filter_1:
        fig.add_trace(go.Scatter(x=energy_valid, y=relative_attenuation_filter_1, mode='lines',
                                 line=dict(color=colour_material_1, width=1.5, dash="dash"), name="Transmission Filter 1"))
    if show_attenuation_plot_filter_1:
        fig.add_trace(go.Scatter(x=energy_valid, y=mass_atten_coeff_1_valid, mode='lines',
                                 line=dict(color=colour_material_1a, width=2, dash="dot"), name="Attenuation Filter 1", yaxis="y2"))
    if show_transmission_plot_filter_2:
        fig.add_trace(go.Scatter(x=energy_valid, y=relative_attenuation_filter_2, mode='lines',
                                 line=dict(color=colour_material_2, width=1.5, dash="dash"), name="Transmission Filter 2"))
    if show_attenuation_plot_filter_2:
        fig.add_trace(go.Scatter(x=energy_valid, y=mass_atten_coeff_2_valid, mode='lines',
                                 line=dict(color=colour_material_2a, width=2, dash="dot"), name="Attenuation Filter 2", yaxis="y2"))
    if show_transmission_plot_filter_3:
        fig.add_trace(go.Scatter(x=energy_valid, y=relative_attenuation_filter_3, mode='lines',
                                 line=dict(color=colour_material_3, width=2, dash="dash"), name="Transmission Filter 3"))
    if show_attenuation_plot_filter_3:
        fig.add_trace(go.Scatter(x=energy_valid, y=mass_atten_coeff_3_valid, mode='lines',
                                 line=dict(color=colour_material_3a, width=1.5, dash="dot"), name="Attenuation Filter 3", yaxis="y2"))

    # AUC annotation
    fig.add_annotation(
        x=0.95, y=1.05,
        text=f"Total Energy = {auc_percentage:.2f}%",
        showarrow=False, xref="paper", yref="paper",
        font=dict(color=selected_colour, size=25, family="sans-serif")
    )

    # scatter overlay
    if show_scatter_plot:
        contrast_map = {
            "royalblue": "orange",
            "deepskyblue": "darkorange",
            "tomato": "blue",
            "magenta": "lime",
            "cyan": "crimson",
            "lightgreen": "darkviolet",
        }
        _ = contrast_map.get(selected_colour, "orange")  # kept for parity; line color is fixed to orange below

        scatter_energy, scatter_flux = calculate_compton_scatter_spectrum(
            energy_valid,
            energy_flux_normalised_filtered,
            scatter_angle_deg,
            scatter_mass_atten,
            scatter_density,
            scatter_thickness,
            scatter_energy_base
        )

        primary_max = np.max(energy_flux_normalised_filtered) if len(energy_flux_normalised_filtered) else 1.0
        scatter_flux_scaled = scatter_flux * (primary_max * (scatter_y_scale / 100.0))

        fig.add_trace(go.Scatter(
            x=scatter_energy,
            y=scatter_flux_scaled,
            mode='lines',
            line=dict(color="orange", width=2, dash="dot"),
            name=f"Compton Scatter @ {scatter_angle_deg}°"
        ))

        # optional HVL for scatter
        # interpolate attenuation to scatter energy grid if needed
        if scatter_energy_base is not None and scatter_mass_atten is not None:
            if len(scatter_mass_atten) != len(scatter_energy):
                mass_atten_interp = interp1d(scatter_energy_base, scatter_mass_atten, bounds_error=False, fill_value="extrapolate")
                scatter_mass_atten_interp = mass_atten_interp(scatter_energy)
            else:
                scatter_mass_atten_interp = scatter_mass_atten

            # If user asked for HVL on scatter line, compute outside and annotate in caller (or add here if you prefer).
            # Left here as in the original center-panel (handled in caller when show_scatter_eff_hvl is True).
            # To keep behaviour identical, we’ll compute/annotate below only when toggle is on.
        # end if

    # layout
    fig.update_layout(
        xaxis=dict(
            title="Photon Energy E (keV)",
            range=[0, tube_voltage_max],
            dtick=10,
            showline=True,
            linewidth=3,
            showgrid=False,
            title_font=dict(size=20),
            tickfont=dict(size=18),
        ),
        yaxis=dict(
            title="Relative Energy Flux Ψ",
            range=[0, y_axis_max],
            dtick=0.1,
            showline=True,
            linewidth=3,
            showgrid=False,
            title_font=dict(size=22),
            tickfont=dict(size=18),
        ),
        yaxis2=dict(
            title="Mass Attenuation Coefficient μ (cm²/g)",
            overlaying='y',
            side='right',
            type='log',
            showgrid=False,
            title_font=dict(size=22),
            tickfont=dict(size=18),
        ),
        showlegend=False,
        template=selected_style,
        width=1300,
        height=720,
        uirevision='constant',
    )

    if show_grid:
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

    return fig
