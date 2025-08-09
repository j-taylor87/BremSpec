# figure.py
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from scipy.interpolate import PchipInterpolator

from utils.calc_utils import (
    add_characteristic_peaks,
    calculate_effective_energy_and_hvl,
    calculate_median_energy,
    calculate_mean_energy,
    calculate_peak_energy,
    calculate_compton_scatter_spectrum,
)

# Tunables for plotting quality vs. speed
_N_BASE_ANCHORS = 2500   # ~how many adaptive "anchor" points to keep (fewer -> faster)
_MAX_PLOT_POINTS = 5000 # upper bound for final drawn points (dynamic target below)


def _smooth_for_plot(x, y, anchors=None, n_base=_N_BASE_ANCHORS, fig_width_px=1300):
    """
    Make a smooth, non-negative curve for plotting from (x,y) without changing physics:
      1) build a dense reference with PCHIP on log(y)
      2) choose ~n_base anchors using a curvature-weighted arc-length in log-space
      3) densify to ~1.2x figure width for anti-aliasing
    Returns (x_plot, y_plot).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = x[m]; y = y[m]
    if x.size < 2: 
        return x, y

    order = np.argsort(x)
    x, y = x[order], y[order]

    # PCHIP on log(y) avoids negatives & ringing
    p_log = PchipInterpolator(x, np.log(np.clip(y, 1e-12, None)))

    # Dense reference for measuring slope/curvature
    xd = np.linspace(x[0], x[-1], max(1024, len(x)))
    yd_log = p_log(xd)
    dy = np.gradient(yd_log, xd)
    d2y = np.gradient(dy, xd)

    # Curvature-weighted arc-length (coeffs tuned for smoothness)
    w = np.sqrt(1.0 + (4.0 * dy)**2 + (18.0 * d2y)**2)
    s = np.cumsum(np.maximum(w, 1e-12))
    s = (s - s[0]) / (s[-1] - s[0])

    # Choose anchors uniformly in this metric
    x_base = np.interp(np.linspace(0, 1, n_base), s, xd)

    # Ensure characteristic-peak energies are honoured
    if anchors is not None and len(anchors):
        x_base = np.unique(np.clip(np.concatenate([x_base, np.asarray(anchors, float)]), x[0], x[-1]))

    y_base = np.exp(p_log(x_base))

    # Final densify roughly to pixel width (anti-alias), capped
    n_plot = int(min(_MAX_PLOT_POINTS, max(1000, fig_width_px * 12 // 10)))
    x_plot = np.linspace(x_base[0], x_base[-1], n_plot)
    y_plot = np.exp(PchipInterpolator(x_base, np.log(np.clip(y_base, 1e-12, None)))(x_plot))

    return x_plot, np.clip(y_plot, 0.0, None)


def build_spectrum_figure(
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
):
    fig = go.Figure()

    # Prepare a smooth plotting curve from the physical result (display only)
    # If characteristic peaks are being shown, include their energies as anchors so the line touches them.
    anchors = []
    if show_characteristic_xray_peaks:
        # We only need the annotations/energies; don't replace the physics arrays.
        _, _, _annotations = add_characteristic_peaks(
            target_material, energy_valid, energy_flux_normalised_filtered, tube_voltage
        )
        anchors = [a["energy"] for a in _annotations if a["energy"] <= tube_voltage]

    plot_x, plot_y = _smooth_for_plot(energy_valid, energy_flux_normalised_filtered, anchors=anchors)

    # base spectrum
    fig.add_trace(go.Scatter(
        x=plot_x, y=plot_y, mode='lines',
        line=dict(color=selected_colour, width=1.8, shape="spline", smoothing=0.6),
        name="Spectrum"
    ))

    # characteristic peak annotations (labels/arrows only)
    if show_characteristic_xray_peaks:
        for ann in _annotations:
            if ann["energy"] <= tube_voltage:
                y_val = float(np.interp(ann["energy"], plot_x, plot_y))
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
    # filled area
    fig.add_trace(go.Scatter(
        x=plot_x, y=plot_y, mode='lines', fill='tozeroy',
        line=dict(color=selected_colour, width=1.8, shape="spline", smoothing=0.6),
        fillcolor=fill_color_rgba, name="Filled Area"
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

    # AUC annotation (unchanged)
    fig.add_annotation(
        x=0.95, y=1.05,
        text=f"Total Energy = {auc_percentage:.2f}%",
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

    # layout
    fig.update_layout(
        xaxis=dict(
            title="Photon Energy E (keV)",
            range=[0, float(x_axis_max)],
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
        height= 574,
        uirevision='constant',
    )

    if show_grid:
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

    return fig
