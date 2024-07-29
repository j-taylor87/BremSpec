import numpy as np

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
    - The thickness of the material is converted from mm to cm within the function for calculation purposes.
    """
    mass_atten_coeff_valid = mass_atten_coeff[energy <= tube_voltage]
    exponent = -mass_atten_coeff_valid * filter_thickness / 10 * density # /10 to convert thickness from mm to cm to work with density in g/cm3
    exponent = np.clip(exponent,-100,100) # clip values to prevent overflow issues
    attenuation_relative = np.exp(exponent)  

    return mass_atten_coeff_valid, attenuation_relative