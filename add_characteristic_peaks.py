import numpy as np

def add_characteristic_peaks(energy, energy_flux_normalised_filtered, energy_char, flux_peaks, tube_voltage):
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
    # Filter out energies and their corresponding flux values above the tube_voltage
    energy_valid = energy[energy <= tube_voltage]
    flux_valid = energy_flux_normalised_filtered[energy <= tube_voltage]

    # Filter out characteristic peak energies above the tube_voltage
    peak_energies_valid = [en for en in energy_char if en <= tube_voltage]
    peak_fluxes_valid = [flux_peaks[i] for i, e in enumerate(energy_char) if e <= tube_voltage]

    # Normalise and adjust peak fluxes
    peak_fluxes_normalised = [flux * max(flux_valid) for flux in peak_fluxes_valid]

    for i, peak_energy in enumerate(peak_energies_valid):
        # Find the closest intensity in the valid Bremsstrahlung spectrum for each peak
        closest_index = np.abs(energy_valid - peak_energy).argmin()
        closest_intensity = flux_valid[closest_index]

        # Ensure peak intensity is not less than the Bremsstrahlung intensity at that energy
        if peak_fluxes_normalised[i] < closest_intensity:
            peak_fluxes_normalised[i] = closest_intensity

    # Combine and sort the valid energy and intensity arrays
    energy_combined = np.append(energy_valid, peak_energies_valid)
    intensity_combined = np.append(flux_valid, peak_fluxes_normalised)
    sorted_indices = np.argsort(energy_combined)

    return energy_combined[sorted_indices], intensity_combined[sorted_indices]

