# figure.py
import numpy as np
import plotly.graph_objects as go
# import matplotlib.colors as mcolors
from scipy.interpolate import PchipInterpolator

from utils.calc_utils import (
    add_characteristic_peaks,
    calculate_effective_energy_and_hvl,
    calculate_median_energy,
    calculate_mean_energy,
    calculate_peak_energy,
    calculate_compton_scatter_spectrum,
)
from ui.ui_options_and_styles import rgba

WIDTH_FIGURE = 1300
HEIGHT_FIGURE = 574

# Tunables for plotting quality vs. speed
MAX_PLOT_POINTS = 5000 # upper bound for final drawn points (dynamic target below)

def _smooth_for_plot(
    x_data,
    y_data,
    anchors=None,
    fig_width_px=1300,
    max_plot_points=MAX_PLOT_POINTS,
):
    """
    Smooth the (x, y) data for plotting using PCHIP on log(y) and return a dense grid.

    - Keeps physics intact (no renormalisation).
    - Uses a uniform sampling grid sized to the figure, and
      **merges the exact anchor x positions** (e.g., characteristic-peak energies)
      so the drawn curve passes through peak tops.

    Returns
    -------
    x_plot : ndarray
    y_plot : ndarray
    """
    # Sanitize & sort
    x_source = np.asarray(x_data, dtype=float)
    y_source = np.asarray(y_data, dtype=float)
    valid = np.isfinite(x_source) & np.isfinite(y_source) & (y_source > 0.0)
    x_source = x_source[valid]
    y_source = y_source[valid]
    if x_source.size < 2:
        return x_source, np.clip(y_source, 0.0, None)

    order = np.argsort(x_source)
    x_source = x_source[order]
    y_source = y_source[order]

    # Interpolator in log-space (avoids negatives & ringing)
    eps = 1e-12
    pchip_log = PchipInterpolator(x_source, np.log(np.clip(y_source, eps, None)))

    # Base uniform grid sized to figure width
    n_plot = int(min(max_plot_points, max(1000, int(fig_width_px * 1.2))))
    x_plot_uniform = np.linspace(x_source[0], x_source[-1], n_plot)

    # Merge exact anchor x's so we sample at characteristic-peak energies
    if anchors is not None and len(anchors):
        anchor_x = np.asarray(anchors, dtype=float)
        anchor_x = anchor_x[(anchor_x >= x_source[0]) & (anchor_x <= x_source[-1])]
        x_plot = np.unique(np.concatenate([x_plot_uniform, anchor_x]))

        # If over the cap, decimate ONLY the uniform points; always keep the anchors
        excess = x_plot.size - max_plot_points
        if excess > 0:
            keep_idx = np.linspace(0, x_plot_uniform.size - 1,
                                   max(1, max_plot_points - anchor_x.size),
                                   dtype=int)
            x_plot = np.unique(np.concatenate([x_plot_uniform[keep_idx], anchor_x]))
            x_plot.sort()
    else:
        x_plot = x_plot_uniform

    y_plot = np.exp(pchip_log(x_plot))
    return x_plot, np.clip(y_plot, 0.0, None)


def build_spectrum_figure(
    modality,
    *,
    # core data
    energy_valid,
    energy_flux_normalised_filtered,
    auc_percentage,
    x_axis_max,
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
    apply_axis_ranges=False,
    plot_energy_override=None,
    plot_flux_override=None,
    peak_annotations=None,
):
    fig = go.Figure()

    # --- Build the plotting spectrum (DISPLAY ONLY) ---
    if (plot_energy_override is not None) and (plot_flux_override is not None):
        # Use the arrays that already have the characteristic peaks inserted
        plot_energy = np.asarray(plot_energy_override, float)
        plot_flux   = np.asarray(plot_flux_override, float)
        annotations = list(peak_annotations or [])
    else:
        # Fallback: derive from the physics arrays (kept for backwards-compat)
        plot_energy = np.asarray(energy_valid, float)
        plot_flux   = np.asarray(energy_flux_normalised_filtered, float)
        annotations = []
        if show_characteristic_xray_peaks:
            # Only do this if the caller didn't provide overrides
            plot_energy, plot_flux, annotations = add_characteristic_peaks(
                target_material, energy_valid, energy_flux_normalised_filtered, tube_voltage
            )

    # Tell the smoother to preserve the inserted peaks
    anchor_energies = [a["energy"] for a in annotations if a["energy"] <= tube_voltage]
    plot_x, plot_y = _smooth_for_plot(plot_energy, plot_flux, anchors=anchor_energies)

    if show_characteristic_xray_peaks:
        # Important: actually merge peaks into the spectrum we draw
        plot_energy, plot_flux, annotations = add_characteristic_peaks(
            target_material, energy_valid, energy_flux_normalised_filtered, tube_voltage
        )
        peak_energies = [a["energy"] for a in annotations if a["energy"] <= tube_voltage]
    else:
        peak_energies = []

    # Smooth ONLY the spectrum we’re going to plot; ensure the grid contains peak energies
    plot_x, plot_y = _smooth_for_plot(plot_energy, plot_flux, anchors=peak_energies)

    # characteristic peak annotations (arrow tip at TRUE peak height)
    if show_characteristic_xray_peaks and annotations:
        for ann in annotations:
            if ann["energy"] <= tube_voltage:
                # Prefer the exact peak height if the caller provided it; otherwise interpolate
                y_tip = float(ann.get("peak", np.interp(ann["energy"], plot_x, plot_y)))
                fig.add_annotation(
                    x=ann["energy"],
                    y=y_tip,
                    text=ann["text"],
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1,
                    arrowwidth=1.2,
                    standoff=6,
                    ax=ann["xytext"][0],
                    ay=ann["xytext"][1],
                    font=dict(size=16),
                )

    # base fill
    fill_rgba = rgba(selected_colour, 0.20)
    fig.add_trace(go.Scatter(
        x=plot_x, y=plot_y,
        mode="lines",
        line=dict(width=0),
        fill="tozeroy",
        fillcolor=fill_rgba,
        hoverinfo="skip",
        name="Filled Area",
    ))

    # spectrum line
    fig.add_trace(go.Scatter(
        x=plot_x, y=plot_y,
        mode="lines",
        line=dict(color=selected_colour, width=1.8),
        name="Spectrum",
    ))
        
    # effective energy + HVL (material 1)
    if show_effective_energy:
        eff_e, t_hvl = calculate_effective_energy_and_hvl(
            energy_valid, energy_flux_normalised_filtered, mass_atten_coeff_1_valid, filter_1_density
        )
        eff_y = float(np.interp(eff_e, plot_x, plot_y))
        fig.add_shape(type="line", x0=eff_e, x1=eff_e, y0=0, y1=eff_y, line=dict(color="blue", width=1.8, dash="dash"))
        fig.add_annotation(
            x=eff_e, y=eff_y * 1.25,
            text=(f"E<sub>eff</sub> = {eff_e:.2f} keV<br>HVL = {t_hvl*10:.2f} mm {filter_1_material}"),
            showarrow=False, font=dict(color="blue", size=18)
        )

    # median / mean / peak lines (heights from the smoothed curve for clean visuals)
    if show_median_energy:
        med_e = calculate_median_energy(energy_valid, energy_flux_normalised_filtered)
        med_y = float(np.interp(med_e, plot_x, plot_y))
        fig.add_shape(type="line", x0=med_e, x1=med_e, y0=0, y1=med_y, line=dict(color="green", width=2, dash="dash"))
        fig.add_annotation(x=med_e, y=med_y / 2, text=f"E<sub>η</sub> = {med_e:.2f} keV", showarrow=False, font=dict(color="green", size=16))

    if show_mean_energy:
        mean_e = calculate_mean_energy(energy_valid, energy_flux_normalised_filtered)
        mean_y = float(np.interp(mean_e, plot_x, plot_y))
        fig.add_shape(type="line", x0=mean_e, x1=mean_e, y0=0, y1=mean_y, line=dict(color="blueviolet", width=3, dash="dot"))
        fig.add_annotation(x=mean_e, y=mean_y / 8, text=f"E<sub>μ</sub> = {mean_e:.2f} keV", showarrow=False, font=dict(color="blueviolet", size=16))

    if show_peak_energy:
        peak_e = calculate_peak_energy(energy_valid, energy_flux_normalised_filtered)
        peak_y = float(np.interp(peak_e, plot_x, plot_y))
        fig.add_shape(type="line", x0=peak_e, x1=peak_e, y0=0, y1=peak_y, line=dict(color="darkorange", width=1.8, dash="dashdot"))
        fig.add_annotation(x=peak_e, y=peak_y * 1.05, text=f"E<sub>p</sub> = {peak_e:.2f} keV", showarrow=False, font=dict(color="orange", size=16))

    # transmission / attenuation overlays (on the original grid)
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

    auc_energy_text = None
    if modality == "Fluoroscopy":
        # For fluoroscopy, we show the total energy per second
        auc_energy_text = f"Total Energy per second = {auc_percentage:.2f}%"
    elif modality == "CT":
        # For CT, we show the total energy
        auc_energy_text = f"Total Energy per rotation = {auc_percentage:.2f}%"
    else:
        # For other modalities, we can use a generic message
        auc_energy_text = f"Total Energy = {auc_percentage:.2f}%"

    # AUC annotation (label changes for fluoro)
    fig.add_annotation(
        x=0.95, y=1.05,
        text=auc_energy_text,
        showarrow=False, xref="paper", yref="paper",
        font=dict(color=selected_colour, size=25, family="sans-serif")
    )

    # scatter overlay (unchanged data; just drawn on top)
    if show_scatter_plot:
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
        # (Optional HVL for scatter remains handled in panel_centre if enabled.)

    # # layout
    # fig.update_layout(
    #     xaxis=dict(
    #         title="Photon Energy E (keV)",
    #         range=[0, float(x_axis_max)],
    #         # dtick=5,
    #         showline=True,
    #         linewidth=3,
    #         showgrid=False,
    #         title_font=dict(size=20),
    #         tickfont=dict(size=18),
    #     ),
    #     yaxis=dict(
    #         title=("Relative Energy Flux Rate Ψ/s" if is_fluoro else "Relative Energy Flux Ψ"),
    #         range=[0, y_axis_max],
    #         # dtick=0.1,
    #         showline=True,
    #         linewidth=3,
    #         showgrid=False,
    #         title_font=dict(size=22),
    #         tickfont=dict(size=18),
    #     ),
    #     yaxis2=dict(
    #         title="Mass Attenuation Coefficient μ (cm²/g)",
    #         overlaying='y',
    #         side='right',
    #         type='log',
    #         showgrid=False,
    #         title_font=dict(size=22),
    #         tickfont=dict(size=18),
    #     ),
    #     showlegend=False,
    #     template=selected_style,
    #     width=WIDTH_FIGURE,
    #     height= HEIGHT_FIGURE,
    #     uirevision='constant',
    # )

    # -------- LAYOUT (make axes persistent) --------

    y_axis_text = None
    if modality == "Fluoroscopy":
        y_axis_text = "Relative Energy Flux Rate Ψ/s"
    elif modality == "CT":
        y_axis_text = "Relative Energy Flux Ψ/rotation"
    else:
        y_axis_text = "Relative Energy Flux Ψ"


    fig.update_layout(
        xaxis=dict(
            title="Photon Energy E (keV)",
            range=[0.0, float(x_axis_max)],            # <-- fixed, no autoscale
            showline=True, linewidth=3, showgrid=False,
            title_font=dict(size=20), tickfont=dict(size=18),
            # uirevision optional here; fixed ranges will dominate
        ),
        yaxis=dict(
            title= y_axis_text,
            range=[0.0, float(y_axis_max)],            # <-- fixed, no autoscale
            showline=True, linewidth=3, showgrid=False,
            title_font=dict(size=22), tickfont=dict(size=18),
        ),
        yaxis2=dict(
            title="Mass Attenuation Coefficient μ (cm²/g)",
            overlaying='y', side='right', type='log', showgrid=False,
            title_font=dict(size=22), tickfont=dict(size=18),
        ),
        showlegend=False,
        template=selected_style,
        width=WIDTH_FIGURE,
        height=HEIGHT_FIGURE,
        uirevision="main-spectrum",                   
    )

    # Show grid if requested
    if show_grid:
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
    
    return fig
