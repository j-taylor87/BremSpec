import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["font.sans-serif"] = "Tahoma"
from scipy.interpolate import interp1d
from scipy.constants import speed_of_light

# Custom functions
from filter_selection_and_input import filter_selection_and_input
from kramers_law import kramers_law
from relative_attenuation_mass_coeff import relative_attenuation_mass_coeff
from calculate_auc_percentage import calculate_auc_percentage
from calculate_median_energy import calculate_median_energy
from calculate_effective_energy_and_hvl import calculate_effective_energy_and_hvl
from add_characteristic_peaks import add_characteristic_peaks

# Set streamlit page to wide mode
st.set_page_config(layout="wide")

# Main function
if __name__ == "__main__":
    st.title("BremSpec")
    st.write("Bremsstrahlung X-ray Spectrum Visualiser") #`Decelerate and Illuminate`
    
    # Create two columns
    col1, col2, col3 = st.columns([1,2.5,0.5])

    # List of available plot styles
    plot_styles = ["classic","bmh","ggplot","Solarize_Light2","dark_background"]

    with col1: # elements in col1will display in the left column
        #st.subheader("Input Parameters")

        # User input for modality
        modality = st.selectbox("Modality", ["General X-ray", "Mammography (WIP)", "Fluoroscopy (WIP)","CT (WIP)"])  # Add more modalities as needed

        # Set factors based on modality
        if modality == "General X-ray":
            tube_voltage_max = 150.0 # kV
            tube_voltage_min = 40.0
            tube_voltage_default = 80.0 
            tube_current_max = 500.0 # mA
            tube_current_min = 1.0
            tube_current_default = 200.0
            exposure_time_max = 1000.0 # ms
            exposure_time_min = 1.0
            exposure_time_default = 500.0
            current_time_product_max = 500.0 # mAs
            current_time_product_min = 0.0
            current_time_product_default = 100.0
            filters = ["Al (Z=13)", "Cu (Z=29)"]
            automatic_mode = "Automatic Exposure Control (AEC) (WIP)"

        elif modality == "Mammography":
            tube_voltage_max = 50.0
            tube_voltage_min = 10.0
            tube_voltage_default = 30.0
            tube_current_max = 100.0
            tube_current_min = 1.0
            tube_current_default = 50.0
            exposure_time_max = 200.0
            exposure_time_min = 1.0
            exposure_time_default = 100.0
            current_time_product_max = 100.0
            current_time_product_min = 1.0
            current_time_product_default = 20.0
            filters = ["Al (Z=13)","Mo (Z=42)", "Rh (Z=45)", "Ag (Z=47)"]
            automatic_mode = "Automatic Exposure Control (AEC) (WIP)"

        elif modality == "Fluoroscopy (WIP)":
            tube_voltage_max = 133.0
            tube_voltage_min = 40.0
            tube_voltage_default = 50.0
            tube_current_max = 500.0
            tube_current_min = 1.0
            tube_current_default = 100.0
            exposure_time_max = 1000.0
            exposure_time_min = 1.0
            exposure_time_default = 0.1
            pulse_width_max = 20.0 # ms
            pulse_width_min = 1.0
            pulse_width_default = 8.0
            filters = ["Al (Z=13)", "Cu (Z=29)"]
            automatic_mode = "Automatic Dose Rate Control (ADRC) (WIP)"

        elif modality == "CT (WIP)":
            tube_voltage_max = 140.0
            tube_voltage_min = 50.0
            tube_voltage_default = 120.0
            tube_current_max = 1000.0
            tube_current_min = 0.0
            tube_current_default = 500.0
            exposure_time_max = 2.0 # Rotation time
            exposure_time_min = 0.0
            exposure_time_default = 0.5
            filters = ["Al (Z=13)", "Cu (Z=29)", "Sn (Z=50)"]
            automatic_mode = "Automatic Exposure Control (AEC) (WIP)"

        # User input for mode
        mode = st.checkbox(automatic_mode)

        if "tube_voltage_old" not in st.session_state:
            st.session_state.tube_voltage_old = tube_voltage_default  # Default tube voltage

        if "current_time_product_old" not in st.session_state:
            st.session_state.current_time_product_old = tube_current_default*exposure_time_default*3  # Default current-time product

        # User input for technique factors based on selected mode
        if mode: # Automatic mode
            
            tube_voltage = st.slider("Tube Voltage (kV)", min_value=int(tube_voltage_min), max_value=int(tube_voltage_max), value=int(tube_voltage_default))
            
            if modality == "CT":
                tube_current = 1/tube_voltage**5.0
                exposure_time = st.slider("Rotation Time (s)", min_value=exposure_time_min, max_value=exposure_time_max, value=exposure_time_default,format="%.2f")
            else:
                 # Calculate the new current-time product
                current_time_product = st.session_state.current_time_product_old*(st.session_state.tube_voltage_old/tube_voltage)**5.0
                current_time_product_display = st.write("Current-Time Product (mAs): ", round(current_time_product,0))
                
                # Update the old values for the next run
                st.session_state.tube_voltage_old = tube_voltage
                st.session_state.current_time_product_old = current_time_product

        else: # Manual mode
            tube_voltage = st.slider("Tube Voltage (kV)", min_value=int(tube_voltage_min), max_value=int(tube_voltage_max), value=int(tube_voltage_default))
            tube_current = st.slider("Tube Current (mA)", min_value=int(tube_current_min), max_value=int(tube_current_max), value=int(tube_current_default))
            if modality == "CT":
                exposure_time = st.slider("Rotation Time (ms)", min_value=exposure_time_min, max_value=exposure_time_max, value=exposure_time_default,format="%.0f")
            else:
                exposure_time = st.slider("Exposure Time (ms)", min_value=exposure_time_min, max_value=exposure_time_max, value=exposure_time_default,format="%.0f")
                current_time_product_display = st.write("Current-Time Product (mAs): ", round(tube_current*exposure_time / 1000,0))

        # Define a base energy array that all materials should conform to
        num_points = 1000 # higher number of points gives smoother plots but takes longer to compute
        energy_base_array = np.linspace(0, 150, num=num_points)  # Example: from 1 to 200 with 200 points

        # User input for filter materials
        mass_atten_coeff_1, filter_1_material, filter_1_density, filter_1_thickness = filter_selection_and_input(energy_base_array, 1, filters)

        # Determine a default value for the second filter that isn't the same as the first
        default_for_second_filter = filters[1] if filter_1_material == filters[0] else filters[0]
        mass_atten_coeff_2, filter_2_material, filter_2_density, filter_2_thickness = filter_selection_and_input(energy_base_array, 2, filters,default=default_for_second_filter)

        # Checkbox for showing charactersistic X-ray peaks
        show_characteristic_xray_peaks = st.checkbox("Show Characteristic X-ray Peaks", value=False)

        # Checkbox for showing the median beam energy
        show_median_energy = st.checkbox("Show Median Beam Energy", value=False)

        # Checkbox for showing the effective beam energy
        show_effective_energy = st.checkbox("Show Effective Beam Energy (WIP)", value=False)

        # User input for target material
        target_material = st.selectbox("Target Material", ["W (Z=74)", "Rh (Z=45)", "Mo (Z=42)"])
        if target_material == "W (Z=74)":
            Z = 74

            # Characteristic x-ray energies for tungsten (W) in keV
            # https://physics.nist.gov/PhysRefData/XrayTrans/Html/search.html
            # https://www.researchgate.net/publication/344795585_Simulation_of_X-Ray_Shielding_Effect_of_Different_Materials_Based_on_MCNP5#pf3
            # KL2, KL3, KM3, KN3, L2M2 (Select few)
            energy_char = np.array([57.98, 59.32, 67.25, 69.10, 8.97])

            # Estimated relative energy flux of characteristic x-ray peaks
            # These values are just crude estimates of the heights of the peaks relative to the maximum energy flux
            flux_peaks = np.array([1.1, 1.3, 0.8, 0.7, 0.1])

        elif target_material == "Rh (Z=45)":
            Z = 45

        elif target_material == "Mo (Z=42)":
            Z = 42


    with col3:
        # Dropdown for selecting plot style
        selected_style = st.selectbox("Select Plot Style", plot_styles)

        # Dropdown for selecting plotting colour
        selected_colour = st.selectbox("Select Plot Colour", ["royalblue","tomato","teal","lightgreen","lightsteelblue","cyan","magenta","gold","darkorange","darkviolet","crimson","deeppink","grey"])

        # Checkbox for turning grid on/off
        show_grid = st.checkbox("Show Grid", value=False)

        # Checkbox for scaling axes with selected kV
        scale_axes_with_kv = st.checkbox('Scale axes with selected kV')
        
        # Set the maximum tube voltage based selected kV or not for scaling the x-axis
        if scale_axes_with_kv:
            tube_voltage_max = tube_voltage
        
        y_axis_max = st.slider('Set maximum y-axis value:', min_value=0.0, max_value=1.0, value=1.0)

    with col2: # elements in col2 will be displayed in the right column

        # Calculate the spectrum and get energy values below the tube voltage
        if mode: # Automatic mode
            energy_valid, energy_flux_normalised = kramers_law(Z, energy_base_array, tube_voltage, tube_voltage, current_time_product=current_time_product,current_time_product_max=current_time_product_max)

        else: # Manual mode
            energy_valid, energy_flux_normalised = kramers_law(Z, energy_base_array, tube_voltage, tube_voltage, tube_current, tube_current_max, exposure_time, exposure_time_max)

        # Calculate the filtered spectrum
        energy_flux_normalised_filtered = energy_flux_normalised * relative_attenuation_mass_coeff(energy_base_array,filter_1_density, filter_1_thickness, mass_atten_coeff_1,tube_voltage) * relative_attenuation_mass_coeff(energy_base_array,filter_2_density, filter_2_thickness, mass_atten_coeff_2,tube_voltage)

        # Calculate the AUC percentage for the filtered spectrum
        auc_percentage = calculate_auc_percentage(energy_flux_normalised_filtered, energy_valid, 0, tube_voltage, tube_voltage_max)

        # Call calculate_median_energy_and_hvl
        median_energy_at_50pct_auc = calculate_median_energy(energy_valid, energy_flux_normalised_filtered)

        # Call calculate_effective_energy_and_hvl
        energy_eff = calculate_effective_energy_and_hvl(energy_valid, energy_flux_normalised_filtered, filter_1_thickness)
        
        ########## Visualise the spectrum ##########
        plt.style.use(selected_style)
        
        # Create a FontProperties object and set the font
        font = FontProperties()
        # font.set_family('Tahoma')

        fig, ax = plt.subplots(figsize=(14, 8),dpi=600)
   
        x_axis_limit = [0, tube_voltage_max] # Max energy is set by the tube voltage

        if show_characteristic_xray_peaks:
            # Add characteristic peaks to the spectrum
            energy_valid, energy_flux_normalised_filtered = add_characteristic_peaks(energy_valid, energy_flux_normalised_filtered, energy_char, flux_peaks, tube_voltage)

            # Plot the spectrum with characteristic peaks
            ax.plot(energy_valid, energy_flux_normalised_filtered,linestyle="-",linewidth=1.5,color=selected_colour)

            # Manually position each annotation
            annotations = [
                {"energy": energy_char[4], "peak": flux_peaks[4], "text": f"{energy_char[4]} keV", "xytext": (-20, 20)}, # L2M2
                {"energy": energy_char[1], "peak": flux_peaks[1], "text": f"{energy_char[1]} keV", "xytext": (20, 10)}, # KL3
                {"energy": energy_char[2], "peak": flux_peaks[2], "text": f"{energy_char[2]} keV", "xytext": (-20, 15)}, # KM3
                {"energy": energy_char[3], "peak": flux_peaks[3], "text": f"{energy_char[3]} keV", "xytext": (15, 0)},  # KN3
                {"energy": energy_char[0], "peak": flux_peaks[0], "text": f"{energy_char[0]} keV", "xytext": (-40, 10)}, # KL2
            ]

            # Annotate each peak
            for ann in annotations:
                if ann["energy"] <= tube_voltage: # Check if the peak energy is below the tube voltage
                    # Find the index where the peak is located in the energy array
                    peak_index = np.where(energy_valid == ann["energy"])[0]
                    if peak_index.size > 0:  # Check if the peak exists in the array
                        peak_index = peak_index[0]
                        y_value = energy_flux_normalised_filtered[peak_index]  # Correct y-value at the peak

                        if y_value < ann["peak"]:  # Only annotate if the peak is taller than the spectrum
                            # Create the annotation
                            ax.annotate(ann["text"],
                                        xy=(ann["energy"], y_value),
                                        xytext=ann["xytext"],
                                        textcoords="offset points",
                                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"),
                                        fontsize=8)
        else:
            # Plot the spectrum without characteristic peaks
            ax.plot(energy_valid, energy_flux_normalised_filtered,linestyle="-",linewidth=1.5,color=selected_colour)

        # Fill underneath the curve
        ax.fill_between(energy_valid, 0, energy_flux_normalised_filtered, color=selected_colour, alpha=0.4)

        if show_median_energy:
            # Add a vertical line for median energy at 50% AUC
            median_index = np.where(energy_valid >= median_energy_at_50pct_auc)[0][0]
            median_height = energy_flux_normalised_filtered[median_index]
            ax.plot([median_energy_at_50pct_auc, median_energy_at_50pct_auc], [0, median_height], color="navy", 
                    linestyle="--", linewidth=0.8, label=f"Median Energy: {median_energy_at_50pct_auc:.2f} keV")
        
            # Add annotation for the median energy
            ax.annotate(f"Median Energy: {median_energy_at_50pct_auc:.2f} keV", color="navy", 
                        xy=(median_energy_at_50pct_auc, median_height / 2),
                        xytext=(68, -20),  # Adjust these values to position your text
                        textcoords="offset points", 
                        ha="center",
                        fontsize=8,
                        fontproperties=font, 
                        #arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"),
            )
        
        if show_effective_energy:
            # Add a vertical line for effective energy
            ax.plot([energy_eff, energy_eff], [0, 1], color="darkgreen", linestyle="--", linewidth=0.8, label=f"Effective Energy: {energy_eff:.2f} keV")
        
            # Add annotation for the effective energy
            ax.annotate(f"Effective Energy: {energy_eff:.2f} keV", color="darkgreen", 
                        xy=(energy_eff, 0.5),
                        xytext=(68, -20),  # Adjust these values to position your text
                        textcoords="offset points", 
                        ha="center",
                        fontsize=8,
                        fontproperties=font, 
                        #arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"),
            )
        
        # # Annotate the AUC percentage on the plot
        ax.annotate(f"Relative AUC: {auc_percentage:.2f}% (of max factors, unfiltered)", 
                    color = "k",
                    xy=(0.64, 0.95), 
                    xycoords="axes fraction", 
                    fontsize=11,
                    fontproperties=font,
                    bbox=dict(boxstyle=None, fc="0.9"))

        ax.set_xlabel("Photon Energy E (keV)", fontsize=14)
        ax.set_ylabel("Relative Energy Flux", fontsize=14)
        ax.set_xlim(x_axis_limit)
        ax.set_ylim([0, y_axis_max])
        ax.set_xticks(np.arange(0, tube_voltage_max+1, 5))
        ax.set_yticks(np.arange(0, y_axis_max+0.05, 0.05))

        #ax.set_title(f"Bremsstrahlung Spectrum for Z={Z}")

        # Set grid based on checkbox
        ax.grid(show_grid)

        st.pyplot(fig)

        
