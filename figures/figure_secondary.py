# figures/figure_secondary.py
# Project: BremSpec
# Author: James Taylor
# Date: October 2023

import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
from utils.calc_utils import (
    kramers_law,
    relative_attenuation_mass_coeff,
    calculate_effective_energy_and_hvl,
)

# EDGE_COLOUR = "royalblue"

# def build_gxr_kap_3d(
#     *,
#     target_material: str,
#     energy_base,
#     kv_now: float,
#     kv_min: float,
#     kv_max: float,
#     mass_atten_1, dens_1: float, thick_1_mm: float,
#     mass_atten_2, dens_2: float, thick_2_mm: float,
#     mass_atten_3, dens_3: float, thick_3_mm: float,
#     field_area_cm2: float,
#     field_area_max_cm2: float,
#     sid_cm: float,
#     n_patch: int = 33,
# ):
#     import numpy as np
#     import plotly.graph_objects as go
#     from utils.calc_utils import kramers_law

#     # -------- 1) Spectrum → output, transmission, KAP ----------
#     kv_now = float(kv_now)
#     energy_base = np.asarray(energy_base, float)

#     # Force AUTO branch (shape only) so we don't require mA/ms here
#     e_valid, flux = kramers_law(
#         target_material, energy_base, kv_now, float(kv_max), float(kv_min),
#         current_time_product=1.0, current_time_product_max=1.0,
#     )

#     # Interpolate mass atten arrays to e_valid (when provided)
#     ebase = energy_base
#     mu1 = np.interp(e_valid, ebase, np.asarray(mass_atten_1, float)) if mass_atten_1 is not None else None
#     mu2 = np.interp(e_valid, ebase, np.asarray(mass_atten_2, float)) if mass_atten_2 is not None else None
#     mu3 = np.interp(e_valid, ebase, np.asarray(mass_atten_3, float)) if mass_atten_3 is not None else None

#     t1_cm = (float(thick_1_mm) or 0.0) / 10.0
#     t2_cm = (float(thick_2_mm) or 0.0) / 10.0
#     t3_cm = (float(thick_3_mm) or 0.0) / 10.0

#     transmission = np.ones_like(e_valid)
#     if mu1 is not None and dens_1 and t1_cm > 0: transmission *= np.exp(-mu1 * float(dens_1) * t1_cm)
#     if mu2 is not None and dens_2 and t2_cm > 0: transmission *= np.exp(-mu2 * float(dens_2) * t2_cm)
#     if mu3 is not None and dens_3 and t3_cm > 0: transmission *= np.exp(-mu3 * float(dens_3) * t3_cm)

#     flux_filtered = flux * transmission
#     output_rel = float(np.trapz(flux_filtered, e_valid))  # relative output per mAs

#     A = max(float(field_area_cm2), 0.0)           # cm²
#     A_max = max(float(field_area_max_cm2), 1.0)   # cm²
#     D = max(float(sid_cm), 1e-6)                  # cm

#     # Relative KAP per mAs (proportional model)
#     kap_rel = output_rel * (A / (D**2))
#     kap_norm_area = output_rel * ((A / A_max) / (D**2))  # normalised to max field area

#     # -------- 2) Solid angle, spherical base geometry ----------
#     # Assume square field with side s => A = s^2
#     s = float(np.sqrt(A)) if A > 0 else 0.0
#     half_a = half_b = s / 2.0

#     # Solid angle of rectangle (±a, ±b) at distance D
#     if A > 0:
#         omega = 4.0 * np.arctan((half_a * half_b) / (D * np.sqrt(D*D + half_a*half_a + half_b*half_b)))
#     else:
#         omega = 0.0
#     spherical_area = omega * (D**2)  # should equal the curved patch area on sphere of radius D

#     # Map plane points (u,v,D) to sphere radius D along rays from origin
#     n_grid = int(max(11, n_patch))
#     u = np.linspace(-half_a, half_a, n_grid)
#     v = np.linspace(-half_b, half_b, n_grid)
#     U, V = np.meshgrid(u, v, indexing="xy")
#     denom = np.sqrt(U*U + V*V + D*D)
#     Xs = D * (U / denom)
#     Ys = D * (V / denom)
#     Zs = D * (D / denom)

#     # Corner directions → points on sphere (r = D)
#     corners_plane = np.array([
#         [-half_a, -half_b, D],
#         [ half_a, -half_b, D],
#         [ half_a,  half_b, D],
#         [-half_a,  half_b, D],
#     ], dtype=float)
#     norms = np.linalg.norm(corners_plane, axis=1, keepdims=True)
#     corners_sphere = D * (corners_plane / np.where(norms == 0, 1.0, norms))

#     # Rays from apex to spherical corners
#     rays = []
#     for p in corners_sphere:
#         rays.append(go.Scatter3d(
#             x=[0.0, p[0]], y=[0.0, p[1]], z=[0.0, p[2]],
#             mode="lines",
#             line=dict(color=EDGE_COLOUR, width=2),   # <- force same colour
#             name="Ray",
#             hoverinfo="skip",
#             opacity=0.9,
#         ))

#     # Great-circle arcs along spherical boundary (adjacent corners)
#     arcs = []
#     m_arc = 150
#     for i in range(4):
#         p = corners_sphere[i]
#         q = corners_sphere[(i + 1) % 4]
#         u_vec = p / np.linalg.norm(p)
#         v_vec = q / np.linalg.norm(q)
#         dot_uv = np.clip(u_vec @ v_vec, -1.0, 1.0)
#         theta = np.arccos(dot_uv)
#         if theta < 1e-9:
#             arc_xyz = np.vstack([p, q]).T
#         else:
#             t_vals = np.linspace(0.0, 1.0, m_arc)
#             s1 = np.sin((1.0 - t_vals) * theta) / np.sin(theta)
#             s2 = np.sin(t_vals * theta) / np.sin(theta)
#             arc_xyz = (s1[:, None] * u_vec + s2[:, None] * v_vec) * D
#             arc_xyz = arc_xyz.T  # (3, m_arc)

#         arcs.append(go.Scatter3d(
#             x=arc_xyz[0], y=arc_xyz[1], z=arc_xyz[2],
#             mode="lines",
#             line=dict(color=EDGE_COLOUR, width=2),   # <- force same colour
#             name="Edge (spherical)",
#             hoverinfo="skip",
#             opacity=0.7,
#         ))

#     # Optional translucent side fans (apex to edge arcs)
#     side_fans = []
#     m_fan = 60
#     for i in range(4):
#         p = corners_sphere[i]
#         q = corners_sphere[(i + 1) % 4]
#         u_vec = p / np.linalg.norm(p)
#         v_vec = q / np.linalg.norm(q)
#         dot_uv = np.clip(u_vec @ v_vec, -1.0, 1.0)
#         theta = np.arccos(dot_uv)
#         if theta < 1e-9:
#             arc_pts = np.stack([p, q], axis=0)
#         else:
#             t_vals = np.linspace(0.0, 1.0, m_fan)
#             s1 = np.sin((1.0 - t_vals) * theta) / np.sin(theta)
#             s2 = np.sin(t_vals * theta) / np.sin(theta)
#             arc_pts = (s1[:, None] * u_vec + s2[:, None] * v_vec) * D  # (m_fan, 3)

#         xs = np.concatenate([[0.0], arc_pts[:, 0]])
#         ys = np.concatenate([[0.0], arc_pts[:, 1]])
#         zs = np.concatenate([[0.0], arc_pts[:, 2]])
#         # Triangles: (0, k, k+1)
#         i_idx = []; j_idx = []; k_idx = []
#         for k in range(1, xs.size - 1):
#             i_idx.append(0); j_idx.append(k); k_idx.append(k + 1)

#         side_fans.append(go.Mesh3d(
#             x=xs, y=ys, z=zs,
#             i=i_idx, j=j_idx, k=k_idx,
#             color="royalblue", opacity=0.12,
#             name="Side", hoverinfo="skip",
#         ))

#     # Curved spherical base (this IS the base)
#     spherical_base = go.Surface(
#         x=Xs, y=Ys, z=Zs,
#         opacity=0.35, showscale=False,
#         colorscale=[[0, "yellow"], [1, "red"]],
#         name="Spherical base (Ω·R²)", hoverinfo="skip",
#     )

#     # -------- 3) Figure assembly + camera from apex (“pointy end”) ----------
#     fig = go.Figure(data=[spherical_base, *side_fans, *arcs, *rays])

#     info_text = (
#         f"KAP (rel./mAs): {kap_rel:.3g}<br>"
#         f"Ω (sr): {omega:.4f}<br>"
#         f"Spherical area Ω·R²: {spherical_area:.1f} cm²<br>"
#         f"Normalised-to-max-area KAP: {kap_norm_area:.3g}"
#     )

#     # --- Camera: diagonal side with slight rotation, closer to the object ---
#     camera = dict(
#         eye=dict(x=2.0, y=1.0, z=-0.1),
#         up=dict(x=0.0, y=0.0, z=0.0), 
#         center=dict(x=0.0, y=0.0, z=0.3), 
#         projection=dict(type="perspective"), 
#     )

#     fig.update_layout(
#         scene=dict(
#             camera=camera,
#             aspectmode="data",
#             xaxis_title="x (cm)", yaxis_title="y (cm)", zaxis_title="z (cm)",
#             xaxis=dict(showgrid=False, zeroline=False),
#             yaxis=dict(showgrid=False, zeroline=False),
#             zaxis=dict(showgrid=False, zeroline=False),
#         ),
#         margin=dict(l=10, r=10, t=40, b=20),
#         title="Relative KAP",
#         annotations=[dict(
#             text=info_text, x=0.01, y=0.99, xref="paper", yref="paper",
#             showarrow=False, align="left", bgcolor="rgba(255,255,255,0.75)"
#         )],
#         showlegend=False,
#         height=320,
#         uirevision="gxr-kap-3d",
#     )
#     return fig

# EDGE_COLOUR = "royalblue"

# def build_gxr_kap_3d(
#     *,
#     target_material: str,
#     energy_base_keV,
#     kv_now: float,
#     kv_min: float,
#     kv_max: float,
#     mas_now: float,
#     mas_max: float,
#     mass_atten_1, dens_1: float, thick_1_mm: float,
#     mass_atten_2, dens_2: float, thick_2_mm: float,
#     mass_atten_3, dens_3: float, thick_3_mm: float,
#     field_area_cm2: float,
#     field_area_max_cm2: float,
#     sid_cm: float,
#     sid_ref_cm: float = 100.0,
#     n_patch: int = 33,
# ):

#     # ---------------- 1) Spectrum → transmission → output (clear names) ----------------
#     energy_base_keV = np.asarray(energy_base_keV, float)

#     # shape-only spectra (normalised internally to max-tech across modality);
#     # we’ll scale by mAs explicitly and form a ratio to the reference
#     energy_now_keV, spectrum_now_norm = kramers_law(
#         target_material=target_material,
#         energy=energy_base_keV,
#         tube_voltage=float(kv_now),
#         tube_voltage_max=float(kv_max),
#         tube_voltage_min=float(kv_min),
#         current_time_product=1.0,
#         current_time_product_max=1.0,
#     )

#     def _interp_to_now(arr):
#         return None if arr is None else np.interp(energy_now_keV, energy_base_keV, np.asarray(arr, float))

#     mu_en_1_cm2_per_g = _interp_to_now(mass_atten_1)
#     mu_en_2_cm2_per_g = _interp_to_now(mass_atten_2)
#     mu_en_3_cm2_per_g = _interp_to_now(mass_atten_3)

#     t1_cm = (float(thick_1_mm) or 0.0) / 10.0
#     t2_cm = (float(thick_2_mm) or 0.0) / 10.0
#     t3_cm = (float(thick_3_mm) or 0.0) / 10.0

#     transmission_now = np.ones_like(energy_now_keV)
#     if mu_en_1_cm2_per_g is not None and dens_1 and t1_cm > 0:
#         transmission_now *= np.exp(-mu_en_1_cm2_per_g * float(dens_1) * t1_cm)
#     if mu_en_2_cm2_per_g is not None and dens_2 and t2_cm > 0:
#         transmission_now *= np.exp(-mu_en_2_cm2_per_g * float(dens_2) * t2_cm)
#     if mu_en_3_cm2_per_g is not None and dens_3 and t3_cm > 0:
#         transmission_now *= np.exp(-mu_en_3_cm2_per_g * float(dens_3) * t3_cm)

#     # scale spectrum by current mAs
#     spectrum_now = spectrum_now_norm * transmission_now * float(mas_now)
#     output_now = float(np.trapz(spectrum_now, energy_now_keV))  # arbitrary units ∝ mAs

#     # geometry
#     field_area_now = float(field_area_cm2)
#     field_area_ref = float(field_area_max_cm2)
#     sid_now_cm = float(sid_cm)

#     # current KAP (relative model ∝ output × A / SID²)
#     kap_now = output_now * (field_area_now / (sid_now_cm ** 2))

#     # ---------------- Reference: unfiltered @ kv_max, mas_max, A_max, same SID ----------------
#     energy_ref_keV, spectrum_ref_norm = kramers_law(
#         target_material=target_material,
#         energy=energy_base_keV,
#         tube_voltage=float(kv_max),
#         tube_voltage_max=float(kv_max),
#         tube_voltage_min=float(kv_min),
#         current_time_product=1.0,
#         current_time_product_max=1.0,
#     )
#     # unfiltered → transmission = 1
#     spectrum_ref = spectrum_ref_norm * float(mas_max)
#     output_ref = float(np.trapz(spectrum_ref, energy_ref_keV))
#     kap_ref = output_ref * (field_area_ref / (sid_now_cm ** 2))

#     relative_kap_percent = (100.0 * kap_now / kap_ref) if kap_ref > 0 else 0.0

#     # ---------------- 2) Faster spherical base geometry (adaptive to field size) ----------------
#     # square field: side s where A = s²
#     side_now = math.sqrt(field_area_now) if field_area_now > 0 else 0.0
#     half_a = half_b = side_now / 2.0

#     # solid angle of rectangle at distance D (sid_now_cm)
#     if field_area_now > 0:
#         omega_sr = 4.0 * math.atan(
#             (half_a * half_b) / (sid_now_cm * math.sqrt(sid_now_cm**2 + half_a**2 + half_b**2))
#         )
#     else:
#         omega_sr = 0.0
#     spherical_area_cm2 = omega_sr * (sid_now_cm ** 2)

#     # mesh resolution scales with (A_now / A_ref) for speed
#     area_frac = 0.0 if field_area_ref <= 0 else math.sqrt(max(field_area_now, 0.0) / field_area_ref)
#     n_grid = int(max(9, min(n_patch, 9 + round((n_patch - 9) * area_frac))))

#     u = np.linspace(-half_a, half_a, n_grid)
#     v = np.linspace(-half_b, half_b, n_grid)
#     U, V = np.meshgrid(u, v, indexing="xy")
#     denom = np.sqrt(U * U + V * V + sid_now_cm * sid_now_cm)
#     Xs = sid_now_cm * (U / denom)
#     Ys = sid_now_cm * (V / denom)
#     Zs = sid_now_cm * (sid_now_cm / denom)

#     # corners on the sphere
#     corners_plane = np.array(
#         [[-half_a, -half_b, sid_now_cm],
#          [ half_a, -half_b, sid_now_cm],
#          [ half_a,  half_b, sid_now_cm],
#          [-half_a,  half_b, sid_now_cm]], dtype=float
#     )
#     norms = np.linalg.norm(corners_plane, axis=1, keepdims=True)
#     corners_sphere = sid_now_cm * (corners_plane / np.where(norms == 0, 1.0, norms))

#     # adaptive detail
#     m_arc = max(24, int(60 * area_frac))
#     m_fan = max(12, int(30 * area_frac))

#     # rays (apex → corners)
#     rays = [
#         go.Scatter3d(
#             x=[0.0, p[0]], y=[0.0, p[1]], z=[0.0, p[2]],
#             mode="lines", line=dict(color=EDGE_COLOUR, width=2),
#             name="Ray", hoverinfo="skip", opacity=0.9
#         ) for p in corners_sphere
#     ]

#     # great-circle arcs on spherical boundary
#     arcs = []
#     for i in range(4):
#         p = corners_sphere[i]; q = corners_sphere[(i + 1) % 4]
#         uvec = p / np.linalg.norm(p); vvec = q / np.linalg.norm(q)
#         dot_uv = np.clip(uvec @ vvec, -1.0, 1.0); theta = np.arccos(dot_uv)
#         if theta < 1e-9:
#             xyz = np.vstack([p, q]).T
#         else:
#             t = np.linspace(0.0, 1.0, m_arc)
#             s1 = np.sin((1.0 - t) * theta) / np.sin(theta)
#             s2 = np.sin(t * theta) / np.sin(theta)
#             xyz = (s1[:, None] * uvec + s2[:, None] * vvec) * sid_now_cm
#             xyz = xyz.T
#         arcs.append(go.Scatter3d(
#             x=xyz[0], y=xyz[1], z=xyz[2],
#             mode="lines", line=dict(color=EDGE_COLOUR, width=2),
#             name="Edge (spherical)", hoverinfo="skip", opacity=0.7
#         ))

#     # translucent side fans (same colour as edges)
#     side_fans = []
#     for i in range(4):
#         p = corners_sphere[i]; q = corners_sphere[(i + 1) % 4]
#         uvec = p / np.linalg.norm(p); vvec = q / np.linalg.norm(q)
#         dot_uv = np.clip(uvec @ vvec, -1.0, 1.0); theta = np.arccos(dot_uv)
#         if theta < 1e-9:
#             arc_pts = np.stack([p, q], axis=0)
#         else:
#             t = np.linspace(0.0, 1.0, m_fan)
#             s1 = np.sin((1.0 - t) * theta) / np.sin(theta)
#             s2 = np.sin(t * theta) / np.sin(theta)
#             arc_pts = (s1[:, None] * uvec + s2[:, None] * vvec) * sid_now_cm

#         xs = np.concatenate([[0.0], arc_pts[:, 0]])
#         ys = np.concatenate([[0.0], arc_pts[:, 1]])
#         zs = np.concatenate([[0.0], arc_pts[:, 2]])

#         ntri = xs.size - 2
#         side_fans.append(go.Mesh3d(
#             x=xs, y=ys, z=zs,
#             i=np.zeros(ntri, dtype=int),
#             j=np.arange(1, ntri + 1, dtype=int),
#             k=np.arange(2, ntri + 2, dtype=int),
#             color=EDGE_COLOUR, opacity=0.12,
#             name="Side", hoverinfo="skip"
#         ))

#     spherical_base = go.Surface(
#         x=Xs, y=Ys, z=Zs,
#         opacity=0.35, showscale=False,
#         colorscale=[[0, "yellow"], [1, "red"]],
#         name="Spherical base (Ω·R²)", hoverinfo="skip",
#     )

#     fig = go.Figure(data=[spherical_base, *side_fans, *arcs, *rays])

#     # ---------------- 3) Layout (your camera kept EXACTLY) ----------------
#     info_text = (
#         f"Relative KAP: {relative_kap_percent:.1f}%<br>"
#         f"Ω (sr): {omega_sr:.4f}<br>"
#         f"Spherical area Ω·R²: {spherical_area_cm2:.1f} cm²"
#     )

#     camera = dict(
#         eye=dict(x=2.0, y=1.0, z=-0.1),
#         up=dict(x=0.0, y=0.0, z=0.0),
#         center=dict(x=0.0, y=0.0, z=0.3),
#         projection=dict(type="perspective"),
#     )

#     fig.update_layout(
#         scene=dict(
#             camera=camera, aspectmode="data",
#             xaxis_title="x (cm)", yaxis_title="y (cm)", zaxis_title="z (cm)",
#             xaxis=dict(showgrid=False, zeroline=False),
#             yaxis=dict(showgrid=False, zeroline=False),
#             zaxis=dict(showgrid=False, zeroline=False),
#         ),
#         margin=dict(l=10, r=10, t=40, b=20),
#         title="Relative KAP (%)",
#         annotations=[dict(
#             text=info_text, x=0.01, y=0.99, xref="paper", yref="paper",
#             showarrow=False, align="left", bgcolor="rgba(255,255,255,0.75)"
#         )],
#         showlegend=False, height=320, uirevision="gxr-kap-3d",
#     )

#     return fig

EDGE_COLOUR = "royalblue"

def build_gxr_kap_3d(
    *,
    target_material: str,
    energy_base_keV,
    kv_now: float,
    kv_min: float,
    kv_max: float,
    mas_now: float,
    mas_max: float,
    mass_atten_1, dens_1: float, thick_1_mm: float,
    mass_atten_2, dens_2: float, thick_2_mm: float,
    mass_atten_3, dens_3: float, thick_3_mm: float,
    field_area_cm2: float,
    field_area_max_cm2: float,
    sid_cm: float,
    # Reference-plane (inverse-square demo)
    show_reference_plane: bool = True,
    reference_plane_z_cm: float | None = None,      # if None, uses sid_cm
    reference_plane_area_cm2: float | None = 100.0, # fixed 100 cm² by default
    reference_z_max_cm: float = 150.0,              # far end for colorbar
    n_patch: int = 33,
) -> go.Figure:

    # ---------------- 1) Spectrum → transmission → output ----------------
    energy_base_keV = np.asarray(energy_base_keV, float)

    energy_keV_now, spectrum_norm_now = kramers_law(
        target_material=target_material,
        energy=energy_base_keV,
        tube_voltage=float(kv_now),
        tube_voltage_max=float(kv_max),
        tube_voltage_min=float(kv_min),
        current_time_product=1.0,
        current_time_product_max=1.0,
    )

    def interp_to_now(arr):
        return None if arr is None else np.interp(energy_keV_now, energy_base_keV, np.asarray(arr, float))

    mu_en_1 = interp_to_now(mass_atten_1)
    mu_en_2 = interp_to_now(mass_atten_2)
    mu_en_3 = interp_to_now(mass_atten_3)

    thickness_1_cm = (float(thick_1_mm) or 0.0) / 10.0
    thickness_2_cm = (float(thick_2_mm) or 0.0) / 10.0
    thickness_3_cm = (float(thick_3_mm) or 0.0) / 10.0

    transmission = np.ones_like(energy_keV_now)
    if mu_en_1 is not None and dens_1 and thickness_1_cm > 0:
        transmission *= np.exp(-mu_en_1 * float(dens_1) * thickness_1_cm)
    if mu_en_2 is not None and dens_2 and thickness_2_cm > 0:
        transmission *= np.exp(-mu_en_2 * float(dens_2) * thickness_2_cm)
    if mu_en_3 is not None and dens_3 and thickness_3_cm > 0:
        transmission *= np.exp(-mu_en_3 * float(dens_3) * thickness_3_cm)

    spectrum_now = spectrum_norm_now * transmission * float(mas_now)
    output_now = float(np.trapz(spectrum_now, energy_keV_now))  # ∝ mAs

    # Geometry / KAP (relative)
    field_area_now_cm2 = float(field_area_cm2)
    field_area_ref_cm2 = float(field_area_max_cm2)
    sid_now_cm = float(sid_cm)

    kap_now = output_now * (field_area_now_cm2 / (sid_now_cm ** 2))

    # Reference @ kv_max, mas_max, A_max, same SID
    _, spectrum_ref_norm = kramers_law(
        target_material=target_material,
        energy=energy_base_keV,
        tube_voltage=float(kv_max),
        tube_voltage_max=float(kv_max),
        tube_voltage_min=float(kv_min),
        current_time_product=1.0,
        current_time_product_max=1.0,
    )
    spectrum_ref = spectrum_ref_norm * float(mas_max)           # unfiltered
    output_ref = float(np.trapz(spectrum_ref, energy_base_keV))
    kap_ref = output_ref * (field_area_ref_cm2 / (sid_now_cm ** 2))
    relative_kap_percent = (100.0 * kap_now / kap_ref) if kap_ref > 0 else 0.0

    # ---------------- 2) Spherical field surface (purple) ----------------
    side_length_cm = math.sqrt(field_area_now_cm2) if field_area_now_cm2 > 0 else 0.0
    half_width_cm = side_length_cm / 2.0

    if field_area_now_cm2 > 0:
        solid_angle_sr = 4.0 * math.atan(
            (half_width_cm * half_width_cm) / (sid_now_cm * math.sqrt(sid_now_cm**2 + 2*half_width_cm**2))
        )
    else:
        solid_angle_sr = 0.0
    spherical_area_cm2 = solid_angle_sr * (sid_now_cm ** 2)

    area_fraction = 0.0 if field_area_ref_cm2 <= 0 else math.sqrt(max(field_area_now_cm2, 0.0) / field_area_ref_cm2)
    grid_n = int(max(9, min(n_patch, 9 + round((n_patch - 9) * area_fraction))))

    x_plane_grid = np.linspace(-half_width_cm, +half_width_cm, grid_n)
    y_plane_grid = np.linspace(-half_width_cm, +half_width_cm, grid_n)
    X_plane_grid, Y_plane_grid = np.meshgrid(x_plane_grid, y_plane_grid, indexing="xy")

    radius_from_source = np.sqrt(X_plane_grid**2 + Y_plane_grid**2 + sid_now_cm**2)
    X_sphere = sid_now_cm * (X_plane_grid / radius_from_source)
    Y_sphere = sid_now_cm * (Y_plane_grid / radius_from_source)
    Z_sphere = sid_now_cm * (sid_now_cm / radius_from_source)

    spherical_surface = go.Surface(
        x=X_sphere, y=Y_sphere, z=Z_sphere,
        surfacecolor=np.zeros_like(X_sphere),
        colorscale=[[0.0, "#57BCFF"], [1.0, "#57BCFF"]],  # constant purple
        showscale=False,
        opacity=0.35,
        name="Field surface",
        hoverinfo="skip",
    )

    # Corner points on the sphere
    corners_plane = np.array(
        [[-half_width_cm, -half_width_cm, sid_now_cm],
         [ half_width_cm, -half_width_cm, sid_now_cm],
         [ half_width_cm,  half_width_cm, sid_now_cm],
         [-half_width_cm,  half_width_cm, sid_now_cm]], dtype=float
    )
    norms = np.linalg.norm(corners_plane, axis=1, keepdims=True)
    corners_sphere = sid_now_cm * (corners_plane / np.where(norms == 0, 1.0, norms))

    # Rays (apex → corners)
    rays = [
        go.Scatter3d(
            x=[0.0, corner_vec[0]], y=[0.0, corner_vec[1]], z=[0.0, corner_vec[2]],
            mode="lines", line=dict(color=EDGE_COLOUR, width=2),
            name="Ray", hoverinfo="skip", opacity=0.9
        )
        for corner_vec in corners_sphere
    ]

    # Great-circle arcs on spherical boundary
    num_arc_points = max(24, int(60 * area_fraction))
    arcs = []
    for i in range(4):
        p_vec = corners_sphere[i]
        q_vec = corners_sphere[(i + 1) % 4]
        u_vec = p_vec / np.linalg.norm(p_vec)
        v_vec = q_vec / np.linalg.norm(q_vec)
        dot_uv = np.clip(u_vec @ v_vec, -1.0, 1.0)
        theta = np.arccos(dot_uv)
        if theta < 1e-9:
            xyz = np.vstack([p_vec, q_vec]).T
        else:
            t = np.linspace(0.0, 1.0, num_arc_points)
            s1 = np.sin((1.0 - t) * theta) / np.sin(theta)
            s2 = np.sin(t * theta) / np.sin(theta)
            xyz = (s1[:, None] * u_vec + s2[:, None] * v_vec) * sid_now_cm
            xyz = xyz.T
        arcs.append(go.Scatter3d(
            x=xyz[0], y=xyz[1], z=xyz[2],
            mode="lines", line=dict(color=EDGE_COLOUR, width=2),
            name="Edge (spherical)", hoverinfo="skip", opacity=0.7
        ))

    # ---------- RESTORED: translucent side fans (pyramidoid shading) ----------
    side_fans = []
    num_fan_points = max(12, int(30 * area_fraction))
    for i in range(4):
        p_vec = corners_sphere[i]
        q_vec = corners_sphere[(i + 1) % 4]
        u_vec = p_vec / np.linalg.norm(p_vec)
        v_vec = q_vec / np.linalg.norm(q_vec)
        dot_uv = np.clip(u_vec @ v_vec, -1.0, 1.0)
        theta = np.arccos(dot_uv)

        if theta < 1e-9:
            arc_points = np.stack([p_vec, q_vec], axis=0)
        else:
            t = np.linspace(0.0, 1.0, num_fan_points)
            s1 = np.sin((1.0 - t) * theta) / np.sin(theta)
            s2 = np.sin(t * theta) / np.sin(theta)
            arc_points = (s1[:, None] * u_vec + s2[:, None] * v_vec) * sid_now_cm  # (N,3)

        x_fan = np.concatenate([[0.0], arc_points[:, 0]])
        y_fan = np.concatenate([[0.0], arc_points[:, 1]])
        z_fan = np.concatenate([[0.0], arc_points[:, 2]])

        triangles_count = x_fan.size - 2
        side_fans.append(go.Mesh3d(
            x=x_fan, y=y_fan, z=z_fan,
            i=np.zeros(triangles_count, dtype=int),
            j=np.arange(1, triangles_count + 1, dtype=int),
            k=np.arange(2, triangles_count + 2, dtype=int),
            color=EDGE_COLOUR, opacity=0.12,
            name="Side", hoverinfo="skip"
        ))
    # --------------------------------------------------------------------------

    # ---------------- 3) Reference plane (inverse-square visual) ----------------
    traces = [spherical_surface, *side_fans, *arcs, *rays]

    if show_reference_plane:
        ref_z_cm = float(reference_plane_z_cm if reference_plane_z_cm is not None else sid_now_cm)
        patch_area_cm2 = float(100.0 if reference_plane_area_cm2 in (None, 0.0) else reference_plane_area_cm2)
        patch_side_cm = math.sqrt(patch_area_cm2)
        patch_half_cm = patch_side_cm / 2.0

        # Plane patch centered on axis at z = ref_z_cm (square 100 cm²)
        n_ref = max(15, grid_n)
        x_ref_vec = np.linspace(-patch_half_cm, patch_half_cm, n_ref)
        y_ref_vec = np.linspace(-patch_half_cm, patch_half_cm, n_ref)
        X_ref_grid, Y_ref_grid = np.meshgrid(x_ref_vec, y_ref_vec, indexing="xy")
        Z_ref_grid = np.full_like(X_ref_grid, ref_z_cm)

        # Near→far gradient: dark-red → red → orange → yellow → sky-blue → blue
        distance_scale = [
            [0.00, "#8B0000"],  # dark red
            [0.10, "#FF0000"],  # red
            [0.35, "#FF7F00"],  # orange
            [0.60, "#FFD700"],  # yellow
            [0.80, "#87CEEB"],  # sky blue
            [1.00, "#0000FF"],  # blue
        ]
        cmin = 0.0
        cmax = max(float(reference_z_max_cm), ref_z_cm)

        reference_plane = go.Surface(
            x=X_ref_grid, y=Y_ref_grid, z=Z_ref_grid,
            surfacecolor=np.full_like(X_ref_grid, ref_z_cm, dtype=float),
            colorscale=distance_scale,
            cmin=cmin, cmax=cmax,
            showscale=True,
            colorbar=dict(
                title=dict(text="Distance from source (cm)", side="top"),
                thickness=12, len=0.6, x=1.02, y=0.5,
                tickformat="~d",
            ),
            opacity=0.95,
            name="Reference patch (100 cm²)",
            hoverinfo="skip",
        )
        traces.append(reference_plane)

        # Outline of the reference patch
        outline = go.Scatter3d(
            x=[-patch_half_cm,  patch_half_cm,  patch_half_cm, -patch_half_cm, -patch_half_cm],
            y=[-patch_half_cm, -patch_half_cm,  patch_half_cm,  patch_half_cm, -patch_half_cm],
            z=[ref_z_cm]*5,
            mode="lines",
            line=dict(color="#333", width=3),
            name="Ref outline",
            hoverinfo="skip",
        )
        traces.append(outline)

    # ---------------- 4) Figure layout ----------------
    info_text = (
        f"Relative KAP: {relative_kap_percent:.1f}%<br>"
        f"Ω (sr): {solid_angle_sr:.4f}<br>"
        f"Spherical area Ω·R²: {spherical_area_cm2:.1f} cm²"
    )

    camera = dict(
        eye=dict(x=2.0, y=1.0, z=-0.1),
        up=dict(x=0.0, y=0.0, z=0.0),
        center=dict(x=0.0, y=0.0, z=0.3),
        projection=dict(type="perspective"),
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            camera=camera, aspectmode="data",
            xaxis_title="x (cm)", yaxis_title="y (cm)", zaxis_title="z (cm)",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False),
        ),
        margin=dict(l=10, r=10, t=40, b=20),
        title="Relative KAP (%)",
        annotations=[dict(
            text=info_text, x=0.01, y=0.99, xref="paper", yref="paper",
            showarrow=False, align="left", bgcolor="rgba(255,255,255,0.75)"
        )],
        showlegend=False, height=320, uirevision="gxr-kap-3d",
    )

    return fig



# ---------- Mammography: K-edge utilisation vs kV ----------
def build_mammo_kedge_utilisation_figure(
    target_material: str,
    filter_1_material: str,
    energy_base: np.ndarray,
    kv_current: float,
    kv_min: float,
    kv_max: float,
    mass_atten_1: np.ndarray, dens_1: float, thick_1: float,
    mass_atten_2: np.ndarray, dens_2: float, thick_2: float,
    mass_atten_3: np.ndarray, dens_3: float, thick_3: float,
    sweep_span_kv: float = 6.0,
    band_keV: float = 2.0,
) -> go.Figure:
    symbol = (filter_1_material or "").split(" ")[0].strip()
    k_edges = {"Mo": 20.0, "Rh": 23.2, "Ag": 25.5}
    eK = k_edges.get(symbol)
    if eK is None:
        return go.Figure().update_layout(
            height=180, margin=dict(l=48, r=24, t=24, b=32),
            title="Select Mo/Rh/Ag filter to view K-edge utilisation",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )

    k0 = float(kv_current)
    lo = max(float(kv_min), k0 - sweep_span_kv)
    hi = min(float(kv_max), k0 + sweep_span_kv)
    kvs = np.linspace(lo, hi, 25)

    frac_vals = []
    for kv in kvs:
        # Force AUTO branch with unity mAs
        e_valid, flux = kramers_law(
            target_material, energy_base, kv, kv_max, kv_min,
            current_time_product=1.0, current_time_product_max=1.0
        )
        _, att1 = relative_attenuation_mass_coeff(energy_base, dens_1, thick_1, mass_atten_1, kv)
        _, att2 = relative_attenuation_mass_coeff(energy_base, dens_2, thick_2, mass_atten_2, kv)
        _, att3 = relative_attenuation_mass_coeff(energy_base, dens_3, thick_3, mass_atten_3, kv)
        y = flux * att1 * att2 * att3

        total = np.trapz(y, e_valid)
        band_mask = (e_valid >= eK) & (e_valid <= eK + band_keV)
        band = np.trapz(y[band_mask], e_valid[band_mask]) if np.any(band_mask) else 0.0
        frac_vals.append((band / total) if total > 0 else 0.0)

    fig = go.Figure(go.Scatter(x=kvs, y=frac_vals, mode="lines+markers",
                               hovertemplate="kV=%{x:.1f}<br>Frac=%{y:.3f}<extra></extra>"))
    fig.add_shape(type="line", x0=k0, x1=k0, y0=min(frac_vals), y1=max(frac_vals),
                  line=dict(width=1.5, dash="dot"))
    fig.add_annotation(x=k0, y=float(np.interp(k0, kvs, frac_vals)),
                       text=f"K-edge {symbol} @ {eK:.1f} keV",
                       showarrow=True, yshift=10)

    fig.update_layout(
        height=220, margin=dict(l=48, r=24, t=40, b=40),
        title=f"K-edge utilisation vs kV (filter: {symbol})",
        xaxis=dict(title="kV"),
        yaxis=dict(title=f"Fraction in [{eK:.1f},{eK+band_keV:.1f}] keV"),
        showlegend=False,
    )
    return fig

# ---------------- Fluoro: pulsed waveform ----------------
def build_fluoro_pulse_waveform(
    pulse_rate_hz: float,
    pulse_width_ms: float,
    total_time_s: float = 1.0,
    title_prefix: str = "Pulsed fluoro",
    peak_ma: float = 1.0,
    y_axis_max_ma: float | None = None,   # NEW: lock Y max (e.g., tube_current_max)
) -> go.Figure:
    r = max(0.0, float(pulse_rate_hz))
    w_ms = max(0.0, float(pulse_width_ms))
    T = float(total_time_s)
    peak = max(0.0, float(peak_ma))

    if r > 0:
        period = 1.0 / r
        w_s = min(w_ms / 1000.0, period)
        n_pulses = int(np.floor(T * r + 1e-12))
        duty = min(r * w_s, 1.0)
    else:
        period = np.inf
        w_s = 1.0 if w_ms > 0 else 0.0
        n_pulses = 0
        duty = 1.0 if w_ms > 0 else 0.0

    total_on_time = duty * T
    avg_ma = peak * duty

    # Build stepwise arrays (0 or peak)
    if duty >= 1.0 - 1e-9:
        t = np.array([0.0, T])
        y = np.array([peak, peak])
    elif r == 0 and w_ms <= 0:
        t = np.array([0.0, T])
        y = np.array([0.0, 0.0])
    else:
        times = [0.0]
        vals = [0.0]
        k = 0
        while True:
            t_start = k * period
            if t_start > T + 1e-12:
                break
            t_end = t_start + w_s
            if t_start <= T:
                times += [t_start, t_start]
                vals  += [vals[-1], peak]
            if t_end <= T:
                times += [t_end, t_end]
                vals  += [peak, 0.0]
            else:
                if t_start < T:
                    times += [T]
                    vals  += [peak]
                break
            k += 1
        if times[-1] < T:
            times.append(T)
            vals.append(0.0)
        t = np.array(times)
        y = np.array(vals)

    fig = go.Figure(
        go.Scatter(
            x=t, y=y, mode="lines", line_shape="hv",
            hovertemplate="t = %{x:.3f} s<br>I = %{y:.3g} mA<extra></extra>"
        )
    )

    # --- lock Y axis to specified max (e.g., tube_current_max) ---
    y_top = float(y_axis_max_ma) if y_axis_max_ma is not None and y_axis_max_ma > 0 else max(peak, 1.0)

    # Build ticks that always include the top value (0 … y_top)
    steps = 5  # gives 6 ticks incl. 0 and y_top (e.g., 0,100,…,500 for 500 mA)
    tickvals = [y_top * i / steps for i in range(steps + 1)]
    ticktext = [f"{v:.0f}" for v in tickvals]

    fig.update_layout(
        height=200,
        margin=dict(l=48, r=24, t=24, b=36),
        title="Pulse mA waveform (1 s)",  # keep title short; left panel already shows settings
        showlegend=False,
        xaxis=dict(title="Time (s)", range=[0, T], tickformat=".1f"),
        yaxis=dict(
            title="Tube current (mA)",
            range=[0, y_top],
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            ticks="outside",
        ),
    )
    return fig

# # ---------------- CT: 3D helix vs axial (pitch + mAs_eff) ----------------
# def build_ct_pitch_helix_figure(
#     tube_current_mA: float,
#     rotation_time_s: float,
#     pitch: float,
#     beam_width_mm: float = 40.0,   # total collimation (mm)
#     radius_mm: float = 250.0,      # illustrative gantry radius
#     n_turns: int = 3,              # show more turns
# ) -> go.Figure:
#     # --- geometry ---
#     p = max(1e-6, float(pitch))                   # avoid div-by-zero
#     BW = float(beam_width_mm)
#     feed_per_rot = p * BW                         # table advance per rotation
#     theta = np.linspace(0, 2*np.pi*n_turns, 900)

#     # Horizontal orientation: x = table axis, (y,z) = gantry plane
#     x_c = (feed_per_rot / (2*np.pi)) * theta
#     y_c = radius_mm * np.cos(theta)
#     z_c = radius_mm * np.sin(theta)

#     # Build a ribbon with physical beam width along the table axis (x)
#     # Two edges: x_c ± BW/2, same y/z centreline
#     v = np.array([-0.5, 0.5]) * BW
#     X = x_c[:, None] + v[None, :]           # (N, 2)
#     Y= np.repeat(y_c[:, None], 2, axis=1)  # (N, 2)
#     Z = np.repeat(z_c[:, None], 2, axis=1)  # (N, 2)

#     # Effective mAs
#     mAs_eff = (float(tube_current_mA) * float(rotation_time_s)) / p

#     fig = go.Figure()

#     # Ribbon surface (actual beam width)
#     fig.add_surface(
#         x=X, y=Y, z=Z,
#         showscale=False,
#         opacity=0.55,
#         contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
#     )

#     # Optional: centreline for reference
#     fig.add_trace(go.Scatter3d(
#         x=x_c, y=y_c, z=z_c, mode="lines",
#         line=dict(width=3),
#         name="Centreline",
#         opacity=0.6,
#     ))

#     # Axial reference ring (pitch≈0) at x=0
#     fig.add_trace(go.Scatter3d(
#         x=np.zeros_like(theta), y=y_c, z=z_c,
#         mode="lines",
#         line=dict(width=2, dash="dash"),
#         opacity=0.35,
#         name="Axial (p≈0)",
#     ))

#     # One-rotation advance indicator (Δx per rotation)
#     fig.add_trace(go.Scatter3d(
#         x=[0, feed_per_rot], y=[0, 0], z=[0, 0],
#         mode="lines+markers+text",
#         line=dict(dash="dot"),
#         marker=dict(size=3),
#         text=[None, f"Δz/rot = {feed_per_rot:.1f} mm"],
#         textposition="top center",
#         showlegend=False,
#     ))

#     DEFAULT_CAMERA = dict(eye=dict(x=1.2, y=1.2, z=2.2))  # used only on first render / manual reset

#     fig.update_layout(
#         height=380,
#         margin=dict(l=10, r=10, t=28, b=10),
#         title=f"CT pitch — mAs_eff = {mAs_eff:.1f}",
#         showlegend=False,
#         # IMPORTANT: give scene its own uirevision so 3D camera persists
#         uirevision="ct-helix-view",
#         scene=dict(
#             uirevision="ct-helix-view",
#             xaxis_title="Table z (mm)",
#             yaxis_title="Gantry x (mm)",
#             zaxis_title="Gantry y (mm)",
#             aspectmode="data",
#             xaxis=dict(showspikes=False),
#             yaxis=dict(showspikes=False),
#             zaxis=dict(showspikes=False),
#         ),
#     )

#     # Only set camera initially or when explicitly resetting
#     if st.session_state.get("_ct_camera_reset", True):
#         fig.update_layout(scene_camera=DEFAULT_CAMERA)
#         st.session_state._ct_camera_reset = False

#     return fig

# ---------------- CT: 3D helix over selected scan length (pitch + mAs_eff) ----------------
def build_ct_pitch_helix_figure(
    tube_current_mA: float,
    rotation_time_s: float,
    pitch: float,
    beam_width_mm: float,        # total collimation (mm)
    radius_mm: float = 250.0,           # illustrative gantry radius
    scan_length_mm: float | None = None,  # draw helix to this length
    points_per_turn: int = 300,         # rendering density
) -> go.Figure:
    """
    Render a helical ribbon whose *axial extent equals the selected scan length*.
    Also annotates Δz/rotation, total rotations across the scan, and mAs_eff.
    """
    # ----- inputs & derived quantities -----
    safe_pitch = max(1e-6, float(pitch))                 # avoid div-by-zero
    beam_width = float(beam_width_mm)
    feed_per_rotation_mm = safe_pitch * beam_width       # table advance per rotation

    # If no scan length provided, fall back to ~6 rotations of helix as before
    if scan_length_mm is None or scan_length_mm <= 0:
        approx_turns = 6.0
        scan_length_mm = approx_turns * feed_per_rotation_mm
    else:
        approx_turns = float(scan_length_mm) / max(1e-6, feed_per_rotation_mm)

    # Effective mAs (standard CT convention)
    effective_mAs = (float(tube_current_mA) * float(rotation_time_s)) / safe_pitch

    # ----- parametric helix (truncate exactly at scan_length_mm) -----
    # x (table) grows linearly with theta: x = (feed_per_rot / 2π) * theta
    # => theta_end for given scan length:
    theta_end = (2.0 * np.pi) * (float(scan_length_mm) / max(1e-6, feed_per_rotation_mm))
    n_samples = max(200, int(points_per_turn * max(1.0, approx_turns)))
    theta = np.linspace(0.0, theta_end, n_samples)

    x_center = (feed_per_rotation_mm / (2.0 * np.pi)) * theta
    y_center = radius_mm * np.cos(theta)
    z_center = radius_mm * np.sin(theta)

    # Build a ribbon by offsetting along x (physical beam width)
    half_bw = 0.5 * beam_width
    X = np.column_stack([x_center - half_bw, x_center + half_bw])  # (N, 2)
    Y = np.column_stack([y_center, y_center])                       # (N, 2)
    Z = np.column_stack([z_center, z_center])                       # (N, 2)

    # ----- figure -----
    fig = go.Figure()

    # Ribbon surface (actual beam width)
    fig.add_surface(
        x=X, y=Y, z=Z,
        showscale=False,
        colorscale=[[0, "#3b82f6"], [1, "#da46ee"]],
        opacity=0.55,
        contours={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
        name="Beam",
        
    )

    # Optional: centreline for reference
    fig.add_trace(go.Scatter3d(
        x=x_center, y=y_center, z=z_center,
        mode="lines",
        line=dict(width=3),
        name="Centreline",
        opacity=0.6,
    ))

    # Axial reference ring (start plane) at x=0
    ring_theta = np.linspace(0.0, 2.0 * np.pi, 200)
    ring_y = radius_mm * np.cos(ring_theta)
    ring_z = radius_mm * np.sin(ring_theta)
    fig.add_trace(go.Scatter3d(
        x=np.zeros_like(ring_theta),
        y=ring_y, z=ring_z,
        mode="lines",
        line=dict(width=2, dash="dash"),
        opacity=0.35,
        name="Start plane",
    ))

    # Axial reference ring (end plane) at x=scan_length_mm
    fig.add_trace(go.Scatter3d(
        x=np.full_like(ring_theta, float(scan_length_mm)),
        y=ring_y, z=ring_z,
        mode="lines",
        line=dict(width=2, dash="dash"),
        opacity=0.35,
        name="End plane",
    ))

    # One-rotation advance indicator
    fig.add_trace(go.Scatter3d(
        x=[0.0, feed_per_rotation_mm], y=[0.0, 0.0], z=[0.0, 0.0],
        mode="lines+markers+text",
        line=dict(dash="dot"),
        marker=dict(size=3),
        text=[None, f"Δz/rot = {feed_per_rotation_mm:.1f} mm"],
        textposition="top center",
        showlegend=False,
    ))

    # Scan-length indicator & rotations label
    rotations_total = float(scan_length_mm) / max(1e-6, feed_per_rotation_mm)
    fig.add_trace(go.Scatter3d(
        x=[0.0, float(scan_length_mm)], y=[-radius_mm*1.05, -radius_mm*1.05], z=[-radius_mm*1.05, -radius_mm*1.05],
        mode="lines+markers+text",
        line=dict(width=4),
        marker=dict(size=4),
        text=[None, f"Scan length = {scan_length_mm:.0f} mm  (~{rotations_total:.1f} rotations)"],
        textposition="top center",
        showlegend=False,
        opacity=0.8,
    ))

    DEFAULT_CAMERA = dict(eye=dict(x=1.2, y=1.2, z=2.2))

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=28, b=10),
        title=f"CT X-ray Field Path",
        showlegend=False,
        uirevision="ct-helix-view",
        scene=dict(
            uirevision="ct-helix-view",
            xaxis_title="Table z (mm)",
            yaxis_title="Gantry x (mm)",
            zaxis_title="Gantry y (mm)",
            aspectmode="data",
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            zaxis=dict(showspikes=False),
        ),
    )

    # Persist camera unless explicitly reset
    if st.session_state.get("_ct_camera_reset", True):
        fig.update_layout(scene_camera=DEFAULT_CAMERA)
        st.session_state._ct_camera_reset = False

    return fig
