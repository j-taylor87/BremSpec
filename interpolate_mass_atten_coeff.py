from scipy.interpolate import interp1d

# Custom functions
from adjust_duplicates import adjust_duplicates

def interpolate_mass_atten_coeff(base_energy_array, energy_array, mass_atten_coeff):
    """
    Interpolates the mass attenuation coefficients for a given array of base energies.

    This function takes an array of base energies and uses linear interpolation to
    find the corresponding mass attenuation coefficients based on a provided array
    of energies and their known mass attenuation coefficients. It handles duplicate
    energy values and extrapolates values for energies not in the provided range.

    Parameters:
    base_energy_array (numpy.ndarray): An array of energies for which the mass 
      attenuation coefficients need to be interpolated.
    energy_array (numpy.ndarray): The array of energies corresponding to the provided 
      mass attenuation coefficients. This array is used as the base for interpolation.
    mass_atten_coeff (numpy.ndarray): An array of known mass attenuation coefficients 
      corresponding to the energies in `energy_array`.

    Returns:
    numpy.ndarray: An array of interpolated mass attenuation coefficients corresponding 
      to the energies in `base_energy_array`.
    """
    # Adjust duplicate values in the energy array for accurate interpolation
    energy = adjust_duplicates(energy_array)

    # Create a linear interpolation function
    interpolate = interp1d(energy, mass_atten_coeff, kind="linear", fill_value="extrapolate")

    # Interpolate the mass attenuation coefficients
    mass_atten_coeff_interpolated = interpolate(base_energy_array)

    return mass_atten_coeff_interpolated