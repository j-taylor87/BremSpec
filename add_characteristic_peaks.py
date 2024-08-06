import numpy as np

def add_characteristic_peaks(target_material, energy, energy_flux_normalised_filtered, tube_voltage):
    """
    Integrate characteristic X-ray peaks into an existing normalised Bremsstrahlung spectrum.

    This function adds specified characteristic peaks to an existing spectrum, normalises their intensities 
    relative to the spectrum"s maximum intensity and selected kV, and then sorts the combined energy and intensity arrays 
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
    
    if target_material == "W (Z=74)":
        Z = 74

        # Characteristic x-ray energies for tungsten (W) in keV (a select few)
        # https://physics.nist.gov/PhysRefData/XrayTrans/Html/search.html
        # https://www.researchgate.net/publication/344795585_Simulation_of_X-Ray_Shielding_Effect_of_Different_Materials_Based_on_MCNP5#pf3
        
        energy_char = np.array([57.98, # KL2
                                59.32, # KL3
                                67.25, # KM3
                                69.10, # KN3
                                # 8.97 # L2M2
                                ]) 


        # Estimated relative energy flux of characteristic x-ray peaks
        # These values are just crude estimates of the heights of the peaks relative to the maximum energy flux
        flux_peaks = np.array([1.1, 1.3, 0.8, 0.7])

                    # Manually position each annotation
        annotations = [
            # {"energy": energy_char[4], "peak": flux_peaks[4], "text": f"{energy_char[4]} keV", "xytext": (-20, 20)}, # L2M2
            {"energy": energy_char[1], "peak": flux_peaks[1], "text": f"{energy_char[1]} keV", "xytext": (20, 10)}, # KL3
            {"energy": energy_char[2], "peak": flux_peaks[2], "text": f"{energy_char[2]} keV", "xytext": (-20, 15)}, # KM3
            {"energy": energy_char[3], "peak": flux_peaks[3], "text": f"{energy_char[3]} keV", "xytext": (15, 0)},  # KN3
            {"energy": energy_char[0], "peak": flux_peaks[0], "text": f"{energy_char[0]} keV", "xytext": (-40, 10)}, # KL2
        ]

    elif target_material == "Rh (Z=45)":
        Z = 45

        energy_char = np.array([20.2, # KL3
                                22.7, # KM2
                                ]) 
        
        flux_peaks = np.array([1.3, 0.8,])

        annotations = [
            {"energy": energy_char[0], "peak": flux_peaks[0], "text": f"{energy_char[0]} keV", "xytext": (20, 10)}, # KL3
            {"energy": energy_char[1], "peak": flux_peaks[0], "text": f"{energy_char[1]} keV", "xytext": (-20, 15)}, # KM2
        ]

    elif target_material == "Mo (Z=42)":
        Z = 42

        energy_char = np.array([17.5, # KL3
                                19.6, # KM2
                                ]) 
        
        flux_peaks = np.array([1.3, 0.8,])

        annotations = [
            {"energy": energy_char[0], "peak": flux_peaks[0], "text": f"{energy_char[0]} keV", "xytext": (20, 10)}, # KL3
            {"energy": energy_char[1], "peak": flux_peaks[0], "text": f"{energy_char[1]} keV", "xytext": (-20, 15)}, # KM2
        ]

    # Filter out energies and their corresponding flux values above the tube_voltage
    energy_valid = energy[energy <= tube_voltage]
    flux_valid = energy_flux_normalised_filtered[energy <= tube_voltage]

    # Filter out characteristic peak energies above the tube_voltage
    peak_energies_valid = [en for en in energy_char if en <= tube_voltage]
    peak_fluxes_valid = [flux_peaks[i] for i, energy in enumerate(energy_char) if energy <= tube_voltage]

    # Normalise and adjust peak fluxes, capping them to the max_peak_flux_cap
    max_peak_flux_cap = 1.0
    peak_fluxes_normalised = [min(flux * max(flux_valid), max_peak_flux_cap) for flux in peak_fluxes_valid]

    # Normalise and adjust peak fluxes
    peak_fluxes_normalised = [flux * max(flux_valid) for flux in peak_fluxes_valid]

    for i, peak_energy in enumerate(peak_energies_valid):
        # Find the closest intensity in the valid Bremsstrahlung spectrum for each peak
        closest_index = np.abs(energy_valid - peak_energy).argmin()
        closest_intensity = flux_valid[closest_index]

        # Ensure peak intensity is not less than the Bremsstrahlung intensity at that energy
        if peak_fluxes_normalised[i] < closest_intensity:
            peak_fluxes_normalised[i] = closest_intensity
        
        # Replace the energy and flux at the closest index
        energy_valid[closest_index] = peak_energy
        flux_valid[closest_index] = max(peak_fluxes_normalised[i], closest_intensity)

    # Sort the arrays
    sorted_indices = np.argsort(energy_valid)
    energy_combined = energy_valid[sorted_indices]
    flux_combined = flux_valid[sorted_indices]

    return energy_combined, flux_combined, annotations

