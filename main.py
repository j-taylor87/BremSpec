import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams["font.family"] = "sans-serif"
# plt.rcParams["font.sans-serif"] = "Tahoma"

# Custom functions
from get_modality_settings import get_modality_settings
from select_attenuation import select_attenuation
from kramers_law import kramers_law
from relative_attenuation_mass_coeff import relative_attenuation_mass_coeff
from calculate_auc_percentage import calculate_auc_percentage
from calculate_median_energy import calculate_median_energy
from calculate_effective_energy_and_hvl import calculate_effective_energy_and_hvl
from add_characteristic_peaks import add_characteristic_peaks

# Set data directory
# data_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# print("Current Working Directory:", os.getcwd())
data_dir = "./Data" # Works with GitHub

# Set streamlit page to wide mode
st.set_page_config(layout="wide")

# Main function
if __name__ == "__main__":
    st.title("BremSpec")
    st.write("Bremsstrahlung X-ray Spectrum Visualiser") #`Decelerate and Illuminate :P`
    
    # Create two columns
    col1, col2, col3 = st.columns([0.7,2.5,0.5])

    # List of available plot styles
    plot_styles = ["classic","bmh","ggplot","Solarize_Light2","dark_background"]

    with col1: # elements in col1will display in the left column
        #st.subheader("Input Parameters")

        # User input for modality
        modality = st.selectbox("Modality", ["General X-ray", "Mammography (WIP)", "Fluoroscopy (WIP)","CT (WIP)"])  # Add more modalities as needed

        # Get settings for the selected modality
        settings = get_modality_settings(modality)

        # Extract settings for use
        tube_voltage_max = settings.get("tube_voltage_max", 0)
        tube_voltage_min = settings.get("tube_voltage_min", 0)
        tube_voltage_default = settings.get("tube_voltage_default", 0)
        tube_current_max = settings.get("tube_current_max", 0)
        tube_current_min = settings.get("tube_current_min", 0)
        tube_current_default = settings.get("tube_current_default", 0)
        exposure_time_max = settings.get("exposure_time_max", 0)
        exposure_time_min = settings.get("exposure_time_min", 0)
        exposure_time_default = settings.get("exposure_time_default", 0)
        current_time_product_max = settings.get("current_time_product_max", 0)
        current_time_product_min = settings.get("current_time_product_min", 0)
        current_time_product_default = settings.get("current_time_product_default", 0)
        filters = settings.get("filters", [])
        automatic_mode = settings.get("automatic_mode", "")

        # filters = ["Al (Z=13)", "Cu (Z=29)", "Mo (Z=42)", "Rh (Z=45)", "Ag (Z=47)"]
        # print(filters)

        # Set base energy array for plotting, needs to match interpolated attenuation data
        energy_base_array = np.linspace(0, tube_voltage_max, 10000) # keV

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
                current_time_product = st.session_state.current_time_product_old*(st.session_state.tube_voltage_old/tube_voltage)**2.0
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

        # Default values for each filter
        default_filter_1 = filters[0]
        default_filter_2 = filters[1]
        default_filter_3 = filters[2]

        # Find the indices of the default values
        default_index_1 = filters.index(default_filter_1) if default_filter_1 in filters else 0
        default_index_2 = filters.index(default_filter_2) if default_filter_2 in filters else 0
        default_index_3 = filters.index(default_filter_3) if default_filter_3 in filters else 0

        # Selection boxes for filters
        filter_material_selection_1 = st.selectbox(f"Filter 1 Material", filters, index=default_index_1, key=f"filter_material_1")
        filter_material_selection_2 = st.selectbox(f"Filter 2 Material", filters, index=default_index_2, key=f"filter_material_2")
        filter_material_selection_3 = st.selectbox(f"Filter 3/Attenuator Material", filters, index=default_index_3, key=f"filter_material_3")

        # User input for filter materials
        mass_atten_coeff_1, filter_1_material, filter_1_density, filter_1_thickness = select_attenuation(1,filter_material_selection_1,data_dir)
        
        # Determine a default value for the second filter that isn't the same as the first
        default_for_second_filter = filters[1] if filter_1_material == filters[0] else filters[0]
        mass_atten_coeff_2, filter_2_material, filter_2_density, filter_2_thickness = select_attenuation(2,filter_material_selection_2,data_dir)

        # Determine a default value for the third filter that isn't the same as the first or second
        default_for_third_filter = filters[2] if filter_1_material == filters[0] else filters[0]
        mass_atten_coeff_3, filter_3_material, filter_3_density, filter_3_thickness = select_attenuation(3,filter_material_selection_3,data_dir)

        # User input for target material
        target_material = st.selectbox("Target Material", ["W (Z=74)", "Rh (Z=45) (WIP)", "Mo (Z=42) (WIP)"])
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
        

        elif target_material == "Rh (Z=45)":
            Z = 45

        elif target_material == "Mo (Z=42)":
            Z = 42


    with col3: # col3 before col2 to define the show grid button
        # Dropdown for selecting plot style
        selected_style = st.selectbox("Select Plot Style", plot_styles)

        # Dropdown for selecting plotting colour
        selected_colour = st.selectbox("Select Plot Colour", ["royalblue","tomato","teal","lightgreen","lightsteelblue","cyan","magenta","gold","darkorange","darkviolet","crimson","deeppink","grey"])

        # Checkbox for turning grid on/off
        show_grid = st.checkbox("Show Grid", value=False)

        # Checkbox for scaling axes with selected kV
        scale_axes_with_kv = st.checkbox('Scale x-axis with selected kV')
        
        # Set the maximum tube voltage based selected kV or not for scaling the x-axis
        if scale_axes_with_kv:
            tube_voltage_max = tube_voltage
        
        y_axis_max = st.slider('Set maximum y-axis value:',min_value=0.001,max_value=1.0,step=0.001,value=1.0)

        # Checkbox for showing charactersistic X-ray peaks
        show_characteristic_xray_peaks = st.checkbox("Show Characteristic X-ray Peaks", value=False)

        # Checkbox for showing the median beam energy
        show_median_energy = st.checkbox("Show Median Beam Energy", value=False)

        # Checkbox for showing the effective beam energy
        show_effective_energy = st.checkbox("Show Effective Beam Energy (WIP)", value=False)

        # Checkbox for showing attenuation plot
        show_attenuation_plot_filter_1= st.checkbox('Show attenuation plot for filter 1')
        show_attenuation_plot_filter_2= st.checkbox('Show attenuation plot for filter 2')
        show_attenuation_plot_filter_3= st.checkbox('Show attenuation plot for filter 3/attenuator')


    with col2: # elements in col2 will be displayed in the right column

        # Calculate the spectrum and get energy values below the tube voltage

        if mode: # Automatic mode
            energy_valid, energy_flux_normalised = kramers_law(Z, energy_base_array, tube_voltage, tube_voltage_max, current_time_product=current_time_product,current_time_product_max=current_time_product_max)

        else: # Manual mode
            energy_valid, energy_flux_normalised = kramers_law(Z, energy_base_array, tube_voltage, tube_voltage_max, tube_current, tube_current_max, exposure_time, exposure_time_max)

        # Calculate the filtered spectrum
        mass_atten_coeff_1_valid, relative_attenuation_filter_1 = relative_attenuation_mass_coeff(energy_base_array,filter_1_density,filter_1_thickness,mass_atten_coeff_1,tube_voltage)
        mass_atten_coeff_2_valid, relative_attenuation_filter_2 = relative_attenuation_mass_coeff(energy_base_array,filter_2_density,filter_2_thickness,mass_atten_coeff_2,tube_voltage)
        mass_atten_coeff_3_valid, relative_attenuation_filter_3 = relative_attenuation_mass_coeff(energy_base_array,filter_3_density,filter_3_thickness,mass_atten_coeff_3,tube_voltage)

        energy_flux_normalised_filtered = energy_flux_normalised * relative_attenuation_filter_1 * relative_attenuation_filter_2 * relative_attenuation_filter_3
        
        # Calculate the AUC percentage for the filtered spectrum
        auc_percentage = calculate_auc_percentage(energy_flux_normalised_filtered, energy_valid, 0, tube_voltage, tube_voltage_max)

        # Call calculate_median_energy_and_hvl
        median_energy_at_50pct_auc = calculate_median_energy(energy_valid, energy_flux_normalised_filtered)

        # Call calculate_effective_energy_and_hvl
        # energy_eff = calculate_effective_energy_and_hvl(energy_valid, energy_flux_normalised_filtered, filter_1_thickness)
        
        ###############################################################################################################################################################
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
                # {"energy": energy_char[4], "peak": flux_peaks[4], "text": f"{energy_char[4]} keV", "xytext": (-20, 20)}, # L2M2
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

        if show_attenuation_plot_filter_1:        
            # ax.plot(energy_valid,mass_atten_coeff_1_valid/np.max(mass_atten_coeff_1_valid),linestyle="--",linewidth=1.5,color=selected_colour)
            ax.plot(energy_valid,relative_attenuation_filter_1,linestyle="--",linewidth=1.5,color='g')

        if show_attenuation_plot_filter_2:        
            # ax.plot(energy_valid,mass_atten_coeff_1_valid/np.max(mass_atten_coeff_1_valid),linestyle="--",linewidth=1.5,color=selected_colour)
            ax.plot(energy_valid,relative_attenuation_filter_2,linestyle="--",linewidth=1.5,color='r')

        if show_attenuation_plot_filter_3:        
            # ax.plot(energy_valid,mass_atten_coeff_1_valid/np.max(mass_atten_coeff_1_valid),linestyle="--",linewidth=1.5,color=selected_colour)
            ax.plot(energy_valid,relative_attenuation_filter_3,linestyle="--",linewidth=1.5,color='violet')

        
        # Annotate the AUC percentage on the plot
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
        ax.set_xticks(np.arange(0, tube_voltage_max+1, 10))
        ax.set_yticks(np.arange(0, y_axis_max+0.05, 0.1))

        #ax.set_title(f"Bremsstrahlung Spectrum for Z={Z}")

        # Set grid based on checkbox
        ax.grid(show_grid)

        st.pyplot(fig)

        
