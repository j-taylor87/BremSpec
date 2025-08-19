import numpy as np
from scipy.constants import speed_of_light
from scipy.interpolate import interp1d
from scipy.optimize import brentq

def add_characteristic_peaks(
    target_material,
    energy,
    energy_flux_normalised_filtered,
    tube_voltage,
    peak_gain: float = 1.0,  # 1.0 = original heights; e.g. 1.6 makes all peaks 60% taller
):
    """
    Integrate characteristic X-ray peaks into an existing normalised Bremsstrahlung spectrum.

    This function adds specified characteristic peaks to an existing spectrum, normalises their intensities 
    relative to the spectrum's maximum intensity and selected kV, and then sorts the combined energy and intensity arrays 
    for consistent plotting. It is particularly useful for visualising the complete spectrum including both 
    the Bremsstrahlung continuum and characteristic radiation peaks.

    Parameters:
    energy (ndarray): An array of energies in the existing spectrum.
    energy_flux_normalised_filtered (ndarray): An array of normalised energy flux values corresponding to "energy".
    energy_char (ndarray): An array of energies where characteristic peaks occur.
    flux_peaks (ndarray): An array of flux values for each characteristic peak.

    Returns:
    tuple of ndarray: A tuple containing two sorted ndarrays. The first array is the combined energy values (including characteristic peaks),
                      and the second array is the corresponding combined and normalised energy flux values.
    """

    # ----- define peaks (preserve your sources/notes/links) -----
    if target_material == "W (Z=74)":
        atomic_number_Z = 74

        # Characteristic x-ray energies for tungsten (W) in keV (a select few)
        # https://physics.nist.gov/PhysRefData/XrayTrans/Html/search.html
        # https://www.researchgate.net/publication/344795585_Simulation_of_X-Ray_Shielding_Effect_of_Different_Materials_Based_on_MCNP5#pf3
        characteristic_energies_keV = np.array([
            57.98,  # KL2
            59.32,  # KL3
            67.25,  # KM3
            69.10,  # KN3
            # 8.97   # L2M2
        ])

        # Estimated relative energy flux of characteristic x-ray peaks
        # These values are just crude estimates of the heights of the peaks relative to the maximum energy flux
        relative_peak_multipliers = np.array([1.2, 1.4, 1.1, 1.01])

        # Manually position each annotation
        annotations_base = [
            # {"energy": characteristic_energies_keV[4], "peak": relative_peak_multipliers[4], "text": f"{characteristic_energies_keV[4]} keV", "xytext": (-20, 20)},  # L2M2
            {"energy": characteristic_energies_keV[0], "peak": relative_peak_multipliers[0], "text": f"<b>KL2:</b> {characteristic_energies_keV[0]} keV", "xytext": (-60, -30)},  # KL2
            {"energy": characteristic_energies_keV[1], "peak": relative_peak_multipliers[1], "text": f"<b>KL3:</b> {characteristic_energies_keV[1]} keV", "xytext": (-30, -50)},  # KL3
            {"energy": characteristic_energies_keV[2], "peak": relative_peak_multipliers[2], "text": f"<b>KM3:</b> {characteristic_energies_keV[2]} keV", "xytext": (30, -40)},   # KM3
            {"energy": characteristic_energies_keV[3], "peak": relative_peak_multipliers[3], "text": f"<b>KN3:</b> {characteristic_energies_keV[3]} keV", "xytext": (60, -20)},   # KN3
        ]

    elif target_material == "Rh (Z=45)":
        atomic_number_Z = 45
        characteristic_energies_keV = np.array([
            20.2,  # KL3
            22.7,  # KM2
        ])
        relative_peak_multipliers = np.array([1.5, 1.1])

        annotations_base = [
            {"energy": characteristic_energies_keV[0], "peak": relative_peak_multipliers[0], "text": f"<b>KL3:</b> {characteristic_energies_keV[0]} keV", "xytext": (-40, -50)},  # KL3
            {"energy": characteristic_energies_keV[1], "peak": relative_peak_multipliers[1], "text": f"<b>KM2:</b> {characteristic_energies_keV[1]} keV", "xytext": (60, -50)},   # KM2
        ]

    elif target_material == "Mo (Z=42)":
        atomic_number_Z = 42
        characteristic_energies_keV = np.array([
            17.5,  # KL3
            19.6,  # KM2
        ])
        relative_peak_multipliers = np.array([1.5, 1.1])

        annotations_base = [
            {"energy": characteristic_energies_keV[0], "peak": relative_peak_multipliers[0], "text": f"<b>KL3:</b> {characteristic_energies_keV[0]} keV", "xytext": (-50, -50)},  # KL3
            {"energy": characteristic_energies_keV[1], "peak": relative_peak_multipliers[1], "text": f"<b>KM2:</b> {characteristic_energies_keV[1]} keV", "xytext": (50, -50)},   # KM2
        ]

    else:
        # no peaks known for this target
        return np.asarray(energy, float), np.asarray(energy_flux_normalised_filtered, float), []

    # ----- truncate to ≤ kV (copy; don't mutate caller arrays) -----
    energy_array = np.asarray(energy, dtype=float)
    flux_array = np.asarray(energy_flux_normalised_filtered, dtype=float)
    within_kv_mask = np.isfinite(energy_array) & np.isfinite(flux_array) & (energy_array <= float(tube_voltage))
    energy_valid = energy_array[within_kv_mask].copy()
    flux_valid = flux_array[within_kv_mask].copy()
    if energy_valid.size < 2:
        return energy_valid, flux_valid, []

    # sort by energy
    sort_order = np.argsort(energy_valid)
    energy_valid = energy_valid[sort_order]
    flux_valid = flux_valid[sort_order]

    # ----- keep only peaks within kV -----
    peaks_within_kv_mask = characteristic_energies_keV <= float(tube_voltage)
    peak_energies_keV = characteristic_energies_keV[peaks_within_kv_mask]
    peak_ratio_multipliers = relative_peak_multipliers[peaks_within_kv_mask]
    if peak_energies_keV.size == 0:
        annotations_filtered = [a for a in annotations_base if a["energy"] <= float(tube_voltage)]
        return energy_valid, flux_valid, annotations_filtered

    # ----- GLOBAL-MAX scaling (uniform across technique factors) -----
    # This preserves the relative ratios EXACTLY as specified in relative_peak_multipliers.
    global_max_flux = float(flux_valid.max()) if flux_valid.size else 1.0
    target_peak_heights = peak_ratio_multipliers * global_max_flux * float(peak_gain)

    # ----- merge: INSERT peak points (no nearest-sample overwrite) -----
    merged_energy = energy_valid.copy()
    merged_flux = flux_valid.copy()

    for peak_energy_keV, target_height in zip(peak_energies_keV, target_peak_heights):
        peak_energy_keV = float(peak_energy_keV)
        target_height = float(target_height)

        # If the exact energy already exists, set it to the target peak value
        exact_match_idx = np.where(np.isclose(merged_energy, peak_energy_keV, rtol=0.0, atol=1e-6))[0]
        if exact_match_idx.size:
            merged_flux[exact_match_idx[0]] = target_height
        else:
            merged_energy = np.append(merged_energy, peak_energy_keV)
            merged_flux = np.append(merged_flux, target_height)

    # final sort
    final_order = np.argsort(merged_energy)
    energy_combined = merged_energy[final_order]
    flux_combined = merged_flux[final_order]

    # ----- build annotations with actual peak heights (≤ kV) -----
    annotations_output = []
    for a in annotations_base:
        if a["energy"] <= float(tube_voltage):
            idx = np.where(np.isclose(energy_combined, float(a["energy"]), rtol=0.0, atol=1e-6))[0]
            peak_height_value = float(flux_combined[idx[0]]) if idx.size else float(
                np.interp(float(a["energy"]), energy_combined, flux_combined)
            )
            annotations_output.append({
                "energy": float(a["energy"]),
                "peak":   peak_height_value,
                "text":   a["text"],
                "xytext": a["xytext"],
            })

    return energy_combined, flux_combined, annotations_output

# def calculate_auc_percentage(energy_flux_normalised_filtered, energy_valid, energy_lower_bound, energy_upper_bound, tube_voltage_max):
#     """
#     Calculate the AUC percentage for a filtered energy spectrum within a specified energy range.

#     Parameters:
#     energy_flux_normalised_filtered (numpy.ndarray): normalised filtered energy flux.
#     energy_valid (numpy.ndarray): Valid energy values.
#     energy_lower_bound (float): Lower bound of the energy range of interest.
#     energy_upper_bound (float): Upper bound of the energy range of interest.
#     tube_voltage_max (float): Maximum tube voltage.

#     Returns:
#     auc_percentage (float): AUC percentage within the specified energy range.
#     """
#     # Indices for the energy range of interest
#     lower_index = np.searchsorted(energy_valid, energy_lower_bound, side="left")
#     upper_index = np.searchsorted(energy_valid, energy_upper_bound, side="right")

#     # Calculate the AUC for the unfiltered spectrum at maximum technique factor values
#     auc_unfiltered = 0.5 * tube_voltage_max * 1.0

#     # Calculate AUC within the specified energy range
#     auc = np.trapezoid(energy_flux_normalised_filtered[lower_index:upper_index], energy_valid[lower_index:upper_index])

#     # Calculate AUC percentage
#     auc_percentage = (auc / auc_unfiltered) * 100

#     return auc_percentage

def calculate_auc_percentage(energy_flux_normalised_filtered,
                             energy_valid,
                             energy_lower_bound,
                             energy_upper_bound,
                             tube_voltage_max):
    """
    Compute AUC% of the normalised, filtered spectrum over [energy_lower_bound, energy_upper_bound].
    The reference (100%) is the area of the max-technique, unfiltered triangular Kramers spectrum
    after peak normalisation: 0.5 * tube_voltage_max.

    Returns:
        float: AUC percentage.
    """

    # ---- sanitize inputs ----
    energy_keV = np.asarray(energy_valid, dtype=float)
    flux_relative = np.asarray(energy_flux_normalised_filtered, dtype=float)

    finite_mask = np.isfinite(energy_keV) & np.isfinite(flux_relative)
    energy_keV, flux_relative = energy_keV[finite_mask], flux_relative[finite_mask]

    if energy_keV.size < 2:
        return 0.0

    # ensure nondecreasing energy grid
    if not np.all(np.diff(energy_keV) >= 0):
        sort_idx = np.argsort(energy_keV)
        energy_keV = energy_keV[sort_idx]
        flux_relative = flux_relative[sort_idx]

    # ---- integration bounds (clamped to data domain) ----
    range_start_keV = max(float(energy_lower_bound), float(energy_keV[0]))
    range_end_keV   = min(float(energy_upper_bound), float(energy_keV[-1]))
    if not (range_end_keV > range_start_keV):
        return 0.0

    left_idx  = int(np.searchsorted(energy_keV, range_start_keV, side="left"))
    right_idx = int(np.searchsorted(energy_keV, range_end_keV,   side="right"))

    # exact endpoint values via interpolation
    flux_at_start = float(np.interp(range_start_keV, energy_keV, flux_relative))
    flux_at_end   = float(np.interp(range_end_keV,   energy_keV, flux_relative))

    # interior slice (may be empty)
    energy_seg = energy_keV[left_idx:right_idx]
    flux_seg   = flux_relative[left_idx:right_idx]

    # prepend/replace start
    if energy_seg.size == 0 or energy_seg[0] > range_start_keV + 1e-12:
        energy_seg = np.insert(energy_seg, 0, range_start_keV)
        flux_seg   = np.insert(flux_seg,   0, flux_at_start)
    else:
        energy_seg[0] = range_start_keV
        flux_seg[0]   = flux_at_start

    # append/replace end
    if energy_seg[-1] < range_end_keV - 1e-12:
        energy_seg = np.append(energy_seg, range_end_keV)
        flux_seg   = np.append(flux_seg,   flux_at_end)
    else:
        energy_seg[-1] = range_end_keV
        flux_seg[-1]   = flux_at_end

    # ---- integrate filtered spectrum on the bounded segment ----
    area_under_curve = float(np.trapezoid(flux_seg, energy_seg))

    # ---- reference area (max-technique, unfiltered triangular spectrum after normalisation) ----
    reference_area = 0.5 * float(tube_voltage_max)
    if reference_area <= 0.0:
        return 0.0

    auc_percentage = (area_under_curve / reference_area) * 100.0
    return auc_percentage


def calculate_effective_energy_and_hvl(energy_valid, spectrum, mu_en_array, density):
    """
    Calculate the effective energy (monoenergetic HVL-matched) for a polychromatic X-ray spectrum.

    Parameters:
        energy_valid: 1D array-like
            Photon energies (keV) for your spectrum, *already sliced* up to tube voltage.
        spectrum: 1D array-like
            Relative intensities of the spectrum (same shape as energy_valid).
        mu_en_array: 1D array-like
            Mass attenuation coefficients (cm²/g) for the filter material, *same shape as energy_valid*.
            (You must have already loaded or interpolated these onto the same energy grid.)
        density: float
            Density (g/cm³) of the filter material (e.g., aluminium = 2.7 g/cm³).

    Returns:
        effective_energy: float (keV)
            The monoenergetic energy whose HVL in this filter material equals that of the polyenergetic beam.
        t_hvl: float (cm)
            The half-value layer thickness for the polychromatic spectrum (in cm).

    Physics/Method:
        1. The HVL for a polyenergetic (real) spectrum is *not* simply the mean/median energy.
        2. The polyenergetic HVL is found by simulating how much total beam is transmitted through different thicknesses of attenuator.
        3. The *effective energy* is then the energy at which a monoenergetic beam would have the same HVL in the same filter material.
        4. This matches the way HVL and effective energy are used in medical physics and radiology standards (ICRU, IAEA, textbooks).

    Computational Reasoning:
        - Brent’s method (`brentq`) is used to numerically solve for the thickness where the *total* (integrated) transmitted spectrum is 50% of the initial value.
        - The monoenergetic HVL is computed for each energy point, allowing a direct lookup of the "effective energy" that matches the polyenergetic HVL.
    """

    # 1. Convert mass attenuation to linear attenuation for each energy.
    #    This gives μ (1/cm) at each energy in your spectrum, accounting for the material's density.
    mu_linear = mu_en_array * density  # shape: (N,)

    # 2. Define the transmission function for the polyenergetic beam.
    #    For a given thickness t, attenuate each energy using exp(-μ * t),
    #    multiply by the original spectrum, and integrate over energy to get total transmitted intensity.
    #    The result is normalised to the initial beam area (total spectrum intensity).
    def transmission_poly(t):
        transmitted = spectrum * np.exp(-mu_linear * t)  # intensity after filter, all energies
        return np.trapezoid(transmitted, energy_valid) / np.trapezoid(spectrum, energy_valid)  # fraction transmitted

    # 3. Use root finding (brentq) to solve for the thickness (t_hvl) where
    #    the transmission drops to 50% (the formal HVL definition).
    #    Why brentq? It is robust for 1D root finding on monotonic, smooth functions like this.
    #    The brackets [1e-6, 10.0] are chosen to cover possible HVLs (in cm); adjust if your beam/filter is non-standard.
    t_hvl = brentq(lambda t: transmission_poly(t) - 0.5, 1e-6, 10.0)

    # 4. For every energy, compute the monoenergetic HVL: t = ln(2)/μ
    #    This array maps each energy to the thickness that would reduce intensity by half
    #    if the beam was monoenergetic at that energy.
    mono_hvl = np.log(2) / mu_linear  # units: cm

    # 5. The "effective energy" is the *energy* at which the monoenergetic HVL equals the polyenergetic HVL.
    #    Interpolate mono_hvl (as x) vs energy (as y) to look up the energy corresponding to t_hvl.
    energy_from_hvl = interp1d(mono_hvl, energy_valid, bounds_error=False, fill_value="extrapolate")
    effective_energy = float(energy_from_hvl(t_hvl))  # keV

    # 6. Return both the effective energy (keV) and the polyenergetic HVL (cm)
    return effective_energy, t_hvl

def calculate_mean_energy(energy_valid, energy_flux_normalised_filtered):

    # Calculate the total flux
    total_flux = np.trapezoid(energy_flux_normalised_filtered, energy_valid)

    # Calculate the weighted mean of the energy values
    mean_energy = np.trapezoid(energy_flux_normalised_filtered * energy_valid, energy_valid) / total_flux

    return mean_energy

def calculate_median_energy(energy_valid, energy_flux_normalised_filtered):

    # Calculate the cumulative sum of the energy fluxes
    cumulative_energy_flux = np.cumsum(energy_flux_normalised_filtered * np.diff(energy_valid, prepend=0))

    # Normalise by the total AUC
    normalised_cumulative_energy_flux = cumulative_energy_flux / np.trapezoid(energy_flux_normalised_filtered, energy_valid)

    # Find the index for median energy
    indices = np.where(normalised_cumulative_energy_flux >= 0.5)[0]
    if len(indices) > 0:
        median_index = indices[0]
        median_energy_at_50pct_auc = energy_valid[median_index]

        return median_energy_at_50pct_auc
    else:
        # Handle the case where no median is found
        return None  # Return None when median is not found
    
def calculate_peak_energy(energy_valid, energy_flux_normalised_filtered):
    """
    Calculate the energy corresponding to the highest energy flux.

    Parameters:
    energy_valid (np.array): Array of energy values.
    energy_flux_normalised_filtered (np.array): Array of normalized energy flux values.

    Returns:
    float: Energy corresponding to the highest flux.
    """
    # Find the index of the maximum energy flux
    peak_index = np.argmax(energy_flux_normalised_filtered)
    
    # Get the corresponding energy value
    peak_energy = energy_valid[peak_index]
    
    return peak_energy

# def kramers_law(target_material, energy, tube_voltage, tube_voltage_max, tube_voltage_min,
#                 tube_current=None, tube_current_max=None,
#                 exposure_time=None, exposure_time_max=None,
#                 current_time_product=None, current_time_product_max=None):

#     # ----- target Z -----
#     if   target_material == "W (Z=74)": Z = 74
#     elif target_material == "Rh (Z=45)": Z = 45
#     elif target_material == "Mo (Z=42)": Z = 42
#     else:  # safe default
#         Z = 74

#     k_l = 1.0
#     energy = np.asarray(energy, float)
#     tube_voltage = float(tube_voltage)
#     energy_valid = energy[energy <= tube_voltage]

#     # ----- choose mode safely -----
#     auto_requested   = (current_time_product is not None)
#     manual_requested = (tube_current is not None) and (exposure_time is not None)

#     if not auto_requested and not manual_requested:
#         # Fallback so legacy callers don’t crash: assume 1 mAs auto
#         current_time_product = 1.0
#         current_time_product_max = 1.0
#         auto_requested = True

#     # ----- compute energy flux -----
#     if auto_requested:
#         current_time_product_mAs     = float(current_time_product)
#         current_time_product_max_mAs = float(current_time_product_max) if current_time_product_max is not None else current_time_product_mAs

#         energy_flux = (
#             (k_l * Z * current_time_product_mAs) / (2.0 * np.pi * speed_of_light)
#         ) * (tube_voltage - energy_valid)

#         energy_flux_max = (
#             (k_l * Z * current_time_product_max_mAs) / (2.0 * np.pi * speed_of_light)
#         ) * (float(tube_voltage_max) - energy_valid)

#     else:
#         tube_current_mA          = float(tube_current)
#         exposure_time_ms         = float(exposure_time)        # ms (your codebase uses ms)
#         tube_current_max_mA      = float(tube_current_max)     if tube_current_max  is not None else tube_current_mA
#         exposure_time_max_ms     = float(exposure_time_max)    if exposure_time_max is not None else exposure_time_ms

#         energy_flux = (
#             (k_l * Z * tube_current_mA * exposure_time_ms / 1000.0) / (2.0 * np.pi * speed_of_light)
#         ) * (tube_voltage - energy_valid)

#         energy_flux_max = (
#             (k_l * Z * tube_current_max_mA * exposure_time_max_ms / 1000.0) / (2.0 * np.pi * speed_of_light)
#         ) * (float(tube_voltage_max) - energy_valid)

#     # ----- normalise and return -----
#     denom = float(np.max(energy_flux_max)) if np.size(energy_flux_max) and np.max(energy_flux_max) > 0 else 1.0
#     energy_flux_normalised = (energy_flux / denom) * (Z / 74.0)  # keep tungsten scaling

#     return energy_valid, energy_flux_normalised

def kramers_law(target_material, energy, tube_voltage, tube_voltage_max, tube_voltage_min,
                tube_current=None, tube_current_max=None,
                exposure_time=None, exposure_time_max=None,
                current_time_product=None, current_time_product_max=None):
    """
    exposure_time is interpreted as **milliseconds** in the manual branch (consistent with your codebase).
    current_time_product is **mAs** in the auto branch.

    Normalises the spectrum to the *maximum-technique, unfiltered* case on the same energy grid so that
    a max-kV, max-mAs, no-filter spectrum integrates to 100% in your AUC calculation.
    """

    # ----- target Z -----
    if   target_material == "W (Z=74)": Z = 74
    elif target_material == "Rh (Z=45)": Z = 45
    elif target_material == "Mo (Z=42)": Z = 42
    else:  # safe default
        Z = 74

    k_l = 1.0
    energy = np.asarray(energy, float)
    tube_voltage = float(tube_voltage)
    energy_valid = energy[energy <= tube_voltage]

    # ----- choose mode safely -----
    auto_requested   = (current_time_product is not None)
    manual_requested = (tube_current is not None) and (exposure_time is not None)

    if not auto_requested and not manual_requested:
        # Fallback so legacy callers don’t crash: assume 1 mAs auto
        current_time_product = 1.0
        current_time_product_max = 1.0
        auto_requested = True

    # ----- compute energy flux -----
    if auto_requested:
        ctp_now = float(current_time_product)
        # Robust fallback: if max is missing/zero, use current value
        if (current_time_product_max is None) or (float(current_time_product_max) <= 0.0):
            ctp_max = ctp_now
        else:
            ctp_max = float(current_time_product_max)

        energy_flux = (
            (k_l * Z * ctp_now) / (2.0 * np.pi * speed_of_light)
        ) * (tube_voltage - energy_valid)

        # "Max-technique" reference uses tube_voltage_max but the SAME energy grid
        energy_flux_max = (
            (k_l * Z * ctp_max) / (2.0 * np.pi * speed_of_light)
        ) * (float(tube_voltage_max) - energy_valid)

    else:
        mA_now = float(tube_current)
        ms_now = float(exposure_time)  # ms

        # Robust fallbacks for max values
        if (tube_current_max is None) or (float(tube_current_max) <= 0.0):
            mA_max = mA_now
        else:
            mA_max = float(tube_current_max)

        if (exposure_time_max is None) or (float(exposure_time_max) <= 0.0):
            ms_max = ms_now
        else:
            ms_max = float(exposure_time_max)

        energy_flux = (
            (k_l * Z * mA_now * ms_now / 1000.0) / (2.0 * np.pi * speed_of_light)
        ) * (tube_voltage - energy_valid)

        energy_flux_max = (
            (k_l * Z * mA_max * ms_max / 1000.0) / (2.0 * np.pi * speed_of_light)
        ) * (float(tube_voltage_max) - energy_valid)

    # ----- normalise and return -----
    denom = float(np.max(energy_flux_max)) if (np.size(energy_flux_max) and np.max(energy_flux_max) > 0.0) else 1.0
    energy_flux_normalised = (energy_flux / denom) * (Z / 74.0)  # keep tungsten scaling

    return energy_valid, energy_flux_normalised


def relative_attenuation_mass_coeff(energy, density, filter_thickness, mass_atten_coeff, tube_voltage):
    """
    Calculate the relative attenuation of radiation through a material based on its mass attenuation coefficient.

    This function computes the relative attenuation of X-rays or gamma rays as they pass through 
    a given material, which is useful in applications like medical imaging and radiation shielding.

    Parameters:
    energy (ndarray): An array of photon energies (in keV) for which the attenuation is to be calculated.
    density (float): The density of the material (in g/cm³).
    thickness (float): The thickness of the material through which the radiation is passing (in mm).
    mass_atten_coeff (ndarray): An array of mass attenuation coefficients (in cm²/g) corresponding to the energies in the "energy" array.
        These coefficients can be obtained from NIST"s XCOM database.
    tube_voltage (float): The maximum voltage setting (in kV) defining the upper limit of the energy range for calculations.

    Returns:
    ndarray
        An array of relative attenuation values corresponding to each energy value. This represents the 
        fraction of radiation intensity that is not attenuated by the material.

    Note:
    - The NIST XCOM calculator (https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html) can be used 
      to obtain mass attenuation coefficients for different materials and energies.
    - The thickness of the material is converted from mm to cm within the function to match the density in g/cm3.
    """
    mass_atten_coeff_valid = mass_atten_coeff[energy <= tube_voltage]
    exponent = -mass_atten_coeff_valid * filter_thickness / 10 * density # /10 to convert thickness from mm to cm to work with density in g/cm3
    exponent = np.clip(exponent,-100,100) # clip values to prevent overflow issues
    attenuation_relative = np.exp(exponent)  

    return mass_atten_coeff_valid, attenuation_relative

def calculate_compton_scatter_spectrum(
    energy_valid, energy_flux, angle_deg,
    scatter_mass_atten, scatter_density, scatter_thickness,
    scatter_energy_base
):
    """
    Calculate the Compton scatter spectrum at a given angle.
    """
    angle_rad = np.radians(angle_deg)
    m_e = 511.0  # keV

    # Mask out zero-energy points (can't scatter from 0 keV!)
    mask = energy_valid > 0
    energy_valid_nonzero = energy_valid[mask]
    energy_flux_nonzero = energy_flux[mask]

    # Compute scattered photon energies (Compton equation)
    scattered_energy_nonzero = energy_valid_nonzero / (1 + (energy_valid_nonzero / m_e) * (1 - np.cos(angle_rad)))

    # Klein-Nishina weighting, safe division
    epsilon = scattered_energy_nonzero / energy_valid_nonzero
    kn_factor = epsilon**2 * (epsilon + (1/epsilon) - np.sin(angle_rad)**2)

    # Build the smoothed scatter spectrum
    scatter_flux_raw = np.zeros_like(energy_valid_nonzero)
    for Ei, fi, Eprime, kf in zip(energy_valid_nonzero, energy_flux_nonzero, scattered_energy_nonzero, kn_factor):
        if Eprime < energy_valid_nonzero[0] or Eprime > energy_valid_nonzero[-1]:
            continue
        # Gaussian smoothing kernel, width = 0.5 keV (change if needed)
        scatter_flux_raw += fi * kf * np.exp(-0.5 * ((energy_valid_nonzero - Eprime) / 0.5)**2)

    # Interpolate attenuation to plotting grid (but only on nonzero energies)
    if (scatter_mass_atten is not None) and (scatter_energy_base is not None):
        mass_atten_interp = interp1d(scatter_energy_base, scatter_mass_atten, bounds_error=False, fill_value="extrapolate")
        scatter_mass_atten_valid = mass_atten_interp(energy_valid_nonzero)
    else:
        scatter_mass_atten_valid = scatter_mass_atten

    exponent = -scatter_mass_atten_valid * scatter_thickness / 10 * scatter_density
    scatter_flux = scatter_flux_raw * np.exp(np.clip(exponent, -100, 100))
    scatter_flux /= np.max(scatter_flux) if np.max(scatter_flux) > 0 else 1.0

    # Place result back into full grid for plotting (fill zeros at 0 keV)
    scatter_flux_full = np.zeros_like(energy_valid)
    scatter_flux_full[mask] = scatter_flux

    return energy_valid, scatter_flux_full

