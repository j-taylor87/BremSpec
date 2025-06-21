import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

def calculate_effective_energy(energy_valid, spectrum, energy_mu, mu_en_array, density):
    """
    Calculate the effective energy of a spectrum (monoenergetic HVL-matched).
    Parameters:
        energy_valid: array-like, energies (keV) for the spectrum
        spectrum: array-like, relative spectrum (same shape as energy_valid)
        energy_mu: array-like, energies (keV) for mu_en_array
        mu_en_array: array-like, mass attenuation coefficients (cm2/g) for attenuator
        density: float, g/cm3, density of attenuator material
    Returns:
        effective_energy (float), t_hvl (float, cm)
    """
    # Interpolate mu to energy_valid
    mu_interp = interp1d(energy_mu, mu_en_array, bounds_error=False, fill_value="extrapolate")
    mu_spec = mu_interp(energy_valid) * density  # now in 1/cm

    # Transmission as a function of thickness (for the full spectrum)
    def transmission_poly(t):
        atten = np.exp(-mu_spec * t)
        num = np.trapz(spectrum * atten, energy_valid)
        denom = np.trapz(spectrum, energy_valid)
        return num / denom

    # Find t_HVL where transmission drops to 0.5
    t_hvl_poly = brentq(lambda t: transmission_poly(t) - 0.5, 1e-6, 10.0)

    # For monoenergetic: t_HVL(E) = ln(2)/μ(E)
    mu_mono = mu_interp(energy_mu) * density  # in 1/cm
    t_hvl_mono = np.log(2) / mu_mono

    # Interpolate energy vs t_hvl_mono to find the effective energy
    eff_energy_interp = interp1d(t_hvl_mono, energy_mu, bounds_error=False, fill_value="extrapolate")
    effective_energy = float(eff_energy_interp(t_hvl_poly))

    return effective_energy, t_hvl_poly
