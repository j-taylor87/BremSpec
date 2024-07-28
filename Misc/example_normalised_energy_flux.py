import numpy as np
import matplotlib.pyplot as plt

# Constants
k_l = 1.0  # Example constant
Z = 29  # Example atomic number for Copper
speed_of_light = 299792458  # m/s

# Example values
tube_voltage = 150  # keV
tube_current = 1.0  # mA
exposure_time = 1.0  # seconds
tube_voltage_max = 120  # keV
tube_current_max = 1.5  # mA
exposure_time_max = 1.2  # seconds

# Energy range
energy_valid = np.linspace(1, tube_voltage, 100)  # Example energy range from 1 keV to tube voltage

# Calculate energy flux
energy_flux = (k_l * Z * tube_current * exposure_time / 1000) / (2.0 * np.pi * speed_of_light) * (tube_voltage - energy_valid)
energy_flux_max = (k_l * Z * tube_current_max * exposure_time_max / 1000) / (2.0 * np.pi * speed_of_light) * (tube_voltage_max - energy_valid)

# Normalise energy flux
energy_flux_normalised = energy_flux / np.max(energy_flux_max)

# Output for inspection
print("Energy Flux Normalised:", energy_flux_normalised)

# Plot energy flux normalised
plt.plot(energy_valid, energy_flux_normalised, label=f'Tube Voltage = {tube_voltage} keV')
plt.xlabel('Energy (keV)')
plt.ylabel('Normalized Energy Flux')
plt.legend()
plt.title('Normalized Energy Flux vs Energy')
plt.show()
