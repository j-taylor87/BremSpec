import numpy as np
from scipy.interpolate import interp1d

def calculate_effective_energy_and_hvl(energy_valid, energy_flux_normalised_filtered, filter_thickness):

    # Al details
    density = 2.7  # g/cm^3
    energy_Al = np.array([1.0,1.5,1.56,1.56,2,2.51,4.01,5.52,7.02,8.53,10.03,11.54,13.04,14.55,16.05,17.56,19.06,20.57,22.07,23.58,25.08,26.59,28.09,29.6,31.1,32.61,34.11,35.62,37.12,38.63,40.13,41.64,43.14,44.65,46.15,47.66,49.16,50.67,52.17,53.68,55.18,56.69,58.19,59.7,61.2,62.71,64.21,65.72,67.22,68.73,70.23,71.74,73.24,74.75,76.25,77.76,79.26,80.77,82.27,83.78,85.28,86.79,88.29,89.8,91.3,92.81,94.31,95.82,97.32,98.83,100.3,101.8,103.3,104.8,106.4,107.9,109.4,110.9,112.4,113.9,115.4,116.9,118.4,119.9,121.4,122.9,124.4,125.9,127.4,128.9,130.4,131.9,133.4,134.9,136.4,138,139.5,141,142.5,144,145.5,147,148.5,150,130.4,131.9,133.4,134.9,136.4,138,139.5,141,142.5,144,145.5,147,148.5,150,200])
    mass_atten_coeff_Al = np.array([1183,400.2,360,3955,2261,1271,356.7,145.1,72.64,41.07,25.43,16.75,11.63,8.372,6.239,4.767,3.734,2.979,2.423,1.999,1.675,1.419,1.218,1.056,0.9255,0.8178,0.7294,0.6552,0.5934,0.5408,0.4963,0.4579,0.4251,0.3965,0.3717,0.35,0.331,0.3141,0.2993,0.286,0.2742,0.2636,0.2542,0.2456,0.2378,0.2308,0.2244,0.2185,0.2132,0.2083,0.2038,0.1997,0.1958,0.1923,0.189,0.1859,0.1831,0.1804,0.1779,0.1755,0.1733,0.1713,0.1693,0.1675,0.1657,0.1641,0.1625,0.161,0.1596,0.1582,0.1569,0.1557,0.1545,0.1533,0.1522,0.1512,0.1502,0.1492,0.1483,0.1474,0.1465,0.1456,0.1448,0.144,0.1432,0.1425,0.1418,0.141,0.1404,0.1397,0.139,0.1384,0.1378,0.1372,0.1366,0.136,0.1354,0.1348,0.1343,0.1338,0.1332,0.1327,0.1322,0.1317,0.139,0.1384,0.1378,0.1372,0.1366,0.136,0.1354,0.1348,0.1343,0.1338,0.1332,0.1327,0.1322,0.1317,0.1188])

    # # Calculate the cumulative sum of the energy fluxes
    # cumulative_energy_flux = np.cumsum(energy_flux_normalised_filtered * np.diff(energy_valid, prepend=0))

    # # Normalise by the total AUC
    # normalised_cumulative_energy_flux = cumulative_energy_flux / np.trapz(energy_flux_normalised_filtered, energy_valid)
      
    # # Find the index for the where the AUC is 50%
    # indices = np.where(normalised_cumulative_energy_flux >= 0.5)[0]

    # Calculate the HVL mass attenuation coefficient for the Al filter
    mass_atten_coeff_eff = -np.log(energy_flux_normalised_filtered) / (filter_thickness * density / 10)

    # Interpolate to find the effective energy
    interpolate_energy = interp1d(mass_atten_coeff_Al, energy_Al, kind='linear', fill_value='extrapolate')
    energy_eff = interpolate_energy(mass_atten_coeff_eff)

    return energy_eff