import streamlit as st
import os
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
# plt.rcParams["font.sans-serif"] = "Tahoma"
import plotly.graph_objects as go

# Custom functions
from get_modality_settings import get_modality_settings
from select_attenuation import select_attenuation
from kramers_law import kramers_law
from relative_attenuation_mass_coeff import relative_attenuation_mass_coeff
from calculate_auc_percentage import calculate_auc_percentage
from calculate_median_energy import calculate_median_energy
from calculate_mean_energy import calculate_mean_energy
from calculate_peak_energy import calculate_peak_energy
# from calculate_effective_energy_and_hvl import calculate_effective_energy_and_hvl
from add_characteristic_peaks import add_characteristic_peaks

# Set data directory
# data_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# print("Current Working Directory:", os.getcwd())
data_dir = "./Data" # Works with GitHub

# Set streamlit page to wide mode
st.set_page_config(layout="wide", 
                   page_icon='☢️',
                #    menu_items={
                #         'Get Help': '',
                #         'Report a bug': "",
                #         'About': "!"}
)

# Current (Oct 24) CSS workaround for edited whitespace of app
st.markdown("""
    <style>
           /* Remove blank space at top and bottom */ 
           .block-container {
               padding-top: 1rem;
               padding-bottom: 1rem;
            }
           
           /* Remove blank space at the center canvas */ 
           .st-emotion-cache-z5fcl4 {
               position: relative;
               top: -62px;
               }
           
           /* Make the toolbar transparent and the content below it clickable */ 
           .st-emotion-cache-18ni7ap {
               pointer-events: none;
               background: rgb(255 255 255 / 0%)
               }
           .st-emotion-cache-zq5wmm {
               pointer-events: auto;
               background: rgb(255 255 255);
               border-radius: 5px;
               }
    </style>
    """, unsafe_allow_html=True)

# Main function
if __name__ == "__main__":
    st.title("BremSpec")
    st.write("Bremsstrahlung X-ray Spectrum Visualiser") #`Decelerate and Illuminate :P`
    
    # Create columns and specify widths
    col1, col2, col3 = st.columns([0.7,2.2,0.6])

    # List of available plot styles
    # plot_styles = ["classic","bmh","ggplot","Solarize_Light2","dark_background"]
    plot_styles = ['ggplot2', 
                   'seaborn', 
                   'simple_white', 
                   'plotly',
                   'plotly_white', 
                   'plotly_dark', 
                   'presentation'
    ]
    

    with col1: # elements in col1will display in the left column
        #st.subheader("Input Parameters")
        with st.popover("More Info"):
            st.markdown(""" - Select an X-ray imaging modality, technique factors (kV, mA, ms), and filter/target materials to see how these factors affect the 
                    shape of the Bremsstrahlung X-ray spectrum.
                    \n - The available energies along the x-axis will depend on the selected modality, and the x-axis can be set to scale with kV.
                    \n - The y-axis represents the relative energy flux: the energy flux of each energy normalised to the maximum energy flux.
                    \n - Scaling, zooming, and panning tools can be found on the top right of the graph when hovering over it with the mouse. "Axis Options" contains related settings.
                    \n - The area under the curve (AUC), representing the normalised total beam intensity across all available 
                    energies, is displayed in the top-right corner of the plot. This is a relative percentage of the unfiltered beam at maximum technique factors of the selected modality, with a tungsten target. 
                    If scaling x-axis with selected kV is selected, then the AUC calculation will take this as the maximum kV.
                    \n - Median, mean, and peak beam energy can be displayed as vertical lines on the graph. Effective energy and HVL are a WIP.
                    \n - The characteristic X-rays of the anode target material can be displayed.
                    \n - Attenuation and tranmission plots can be viewed for the different filter materials. This is especially useful for visualising absorption edges.""")
        
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
        # energy_base_array = np.linspace(0, tube_voltage_max, 10000) # keV

        # User input for mode
        mode = st.toggle(automatic_mode,value=False)

        if "tube_voltage_old" not in st.session_state:
            st.session_state.tube_voltage_old = tube_voltage_default  # Default tube voltage

        if "current_time_product_old" not in st.session_state:
            st.session_state.current_time_product_old = tube_current_default*exposure_time_default / 1000.0  # Default current-time product
           
        # User input for technique factors based on selected mode
        if mode: # Automatic mode
            tube_voltage = st.slider("Tube Voltage (kV)", min_value=int(tube_voltage_min), max_value=int(tube_voltage_max), value=int(tube_voltage_default))
            
            if modality == "CT (WIP)":
                tube_current = 1/tube_voltage**5.0
                exposure_time = st.slider("Rotation Time (s)", min_value=exposure_time_min, max_value=exposure_time_max, value=exposure_time_default,format="%.2f")
                
                # Calculate the new current-time product
                current_time_product = st.session_state.current_time_product_old*(st.session_state.tube_voltage_old / tube_voltage) ** 2.0
                current_time_product_display = st.write("Current-Time Product (mAs): ", round(current_time_product,0))
                
                # Update the old values for the next run
                st.session_state.tube_voltage_old = tube_voltage
                st.session_state.current_time_product_old = current_time_product
                
            elif modality == "Mammography (WIP)":
                
                # st.session_state.current_time_product_old = 120.0
                # st.session_state.tube_voltage_old = 26.0
                
                # Calculate the new current-time product
                current_time_product = st.session_state.current_time_product_old*(st.session_state.tube_voltage_old / tube_voltage) ** 2.0
                current_time_product_display = st.write("Current-Time Product (mAs): ", round(current_time_product,0))
                
                # Update the old values for the next run
                st.session_state.tube_voltage_old = tube_voltage
                st.session_state.current_time_product_old = current_time_product

            else: # e.g. General X-ray

                # st.session_state.current_time_product_old = 120.0
                # st.session_state.tube_voltage_old = 26.0

                # Calculate the new current-time product
                current_time_product = st.session_state.current_time_product_old*(st.session_state.tube_voltage_old / tube_voltage) ** 2.0
                current_time_product_display = st.write("Current-Time Product (mAs): ", round(current_time_product,0))
                
                # Update the old values for the next run
                st.session_state.tube_voltage_old = tube_voltage
                st.session_state.current_time_product_old = current_time_product

        else: # Manual mode
            tube_voltage = st.slider("Tube Voltage (kV)", min_value=int(tube_voltage_min), max_value=int(tube_voltage_max), value=int(tube_voltage_default))
            tube_current = st.slider("Tube Current (mA)", min_value=int(tube_current_min), max_value=int(tube_current_max), value=int(tube_current_default))
            
            if modality == "CT (WIP)":
                exposure_time = st.slider("Rotation Time (ms)", min_value=exposure_time_min, max_value=exposure_time_max, value=exposure_time_default,format="%.0f")
            elif modality == "Fluoroscopy (WIP)":
                exposure_time = st.slider("Pulse Width (ms)", min_value=exposure_time_min, max_value=exposure_time_max, value=exposure_time_default,format="%.0f")
            else:
                exposure_time = st.slider("Exposure Time (ms)", min_value=exposure_time_min, max_value=exposure_time_max, value=exposure_time_default,format="%.0f")
                # current_time_product_display = st.write("Current-Time Product (mAs): ", round(tube_current*exposure_time / 1000,0))
                
                font_family = "Arial, Helvetica, sans-serif"
                font_size = "17px"  # Specify font size
                number_color = "#FF5733"  # Color for the number (e.g., a shade of orange)

                current_time_product = round(tube_current * exposure_time / 1000)

                current_time_product_display = st.markdown(
                    f"<p style='font-family:{font_family}; font-size:{font_size};'>Current-Time Product (mAs): <span style='color:{number_color};'><b>{current_time_product}</b></span></p>",
                    unsafe_allow_html=True
                )
                
        # User input for target material
        target_material = st.selectbox("Anode Target Material", ["W (Z=74)", "Rh (Z=45)", "Mo (Z=42)"])

        # Checkbox for showing charactersistic X-ray peaks
        show_characteristic_xray_peaks = st.checkbox("Show Characteristic X-ray Peaks", value=False)

        # Checkbox for showing the median beam energy
        show_median_energy = st.checkbox("Show Median Energy Eη", value=False)

        # Checkbox for showing the mean beam energy
        show_mean_energy = st.checkbox("Show Mean Energy Eμ", value=False)

        # Checkbox for showing the mean beam energy
        show_peak_energy = st.checkbox("Show Peak Energy Ep", value=False)

        # # Checkbox for showing the effective beam energy
        # show_effective_energy = st.checkbox("Show Effective Beam Energy (WIP)", value=False)

        # Create two columns: one for all checkboxes and one for the corresponding labels that need markdown to display subscripts
        # col1a, col1b = st.columns([1, 15])
        
        # with col1a:
        #     # All checkboxes in the first column
        #     show_median_energy = st.checkbox("", value=False, key="median_energy")
        #     show_mean_energy = st.checkbox("", value=False, key="mean_energy")
        #     show_peak_energy = st.checkbox("", value=False, key="peak_energy")
        
        # with col1b:
        #     # All labels in the second column
        #     st.markdown(r"Show Median Energy $E_\eta$")
        #     st.markdown(r"Show Mean Energy $E_\mu$")
        #     st.markdown(r"Show Peak Energy $E_{\mathrm{peak}}$")

        # # Create separate columns for each checkbox-label pair
        # col1a, col1b = st.columns([1, 15])
        # with col1a:
        #     show_median_energy = st.checkbox("", value=False, key="median_energy")
        # with col1b:
        #     st.markdown(r"Show Median Energy $E_\eta$")
        
        # col2a, col2b = st.columns([1, 15])
        # with col2a:
        #     show_mean_energy = st.checkbox("", value=False, key="mean_energy")
        # with col2b:
        #     st.markdown(r"Show Mean Energy $E_\mu$")
        
        # col3a, col3b = st.columns([1, 15])
        # with col3a:
        #     show_peak_energy = st.checkbox("", value=False, key="peak_energy")
        # with col3b:
        #     st.markdown(r"Show Peak Energy $E_{\mathrm{peak}}$")

    with col3: # col3 before col2 to define the show grid button    
                
        # Default values for each filter
        default_filter_1 = filters[0]
        default_filter_2 = filters[1]
        default_filter_3 = filters[2]

        # Find the indices of the default values
        default_index_1 = filters.index(default_filter_1) if default_filter_1 in filters else 0
        default_index_2 = filters.index(default_filter_2) if default_filter_2 in filters else 0
        default_index_3 = filters.index(default_filter_3) if default_filter_3 in filters else 0

        # Colours for each material
        colour_material_1 = 'red'
        colour_material_1a = 'orange'
        colour_material_2 = 'green'
        colour_material_2a = 'limegreen'
        colour_material_3 = 'violet'
        colour_material_3a = 'fuchsia'

        with st.container():
            with st.popover("Filter/Attenuator Materials"):
                # Selection boxes for filters
                filter_material_selection_1 = st.selectbox(f"Material 1", filters, index=default_index_1, key=f"filter_material_1")
                filter_material_selection_2 = st.selectbox(f"Material 2", filters, index=default_index_2, key=f"filter_material_2")
                filter_material_selection_3 = st.selectbox(f"Material 3", filters, index=default_index_3, key=f"filter_material_3")
        
        # with st.container():
        #     with st.popover("Tranmission/Attenuation Plots"):
        #         # Checkbox for showing attenuation/transmission plots
        #         st.text("Material 1")
        #         show_transmission_plot_filter_1= st.checkbox('Transmission 1')
        #         show_attenuation_plot_filter_1= st.checkbox('Attenuation 1')
        #         st.text("Material 2")
        #         show_transmission_plot_filter_2= st.checkbox('Transmission 2')
        #         show_attenuation_plot_filter_2= st.checkbox('Attenuation 2')
        #         st.text("Material 3")
        #         show_transmission_plot_filter_3= st.checkbox('Transmission 3')
        #         show_attenuation_plot_filter_3= st.checkbox('Attenuation 3')

        with st.container():
            with st.popover("Transmission/Attenuation Plots"):
                # Display the text in color above each checkbox
                st.markdown(f"<span style='color:{colour_material_1};'>Material 1</span>", unsafe_allow_html=True)
                show_transmission_plot_filter_1 = st.checkbox('Transmission 1')
                show_attenuation_plot_filter_1 = st.checkbox('Attenuation 1')
                
                st.markdown(f"<span style='color:{colour_material_2};'>Material 2</span>", unsafe_allow_html=True)
                show_transmission_plot_filter_2 = st.checkbox('Transmission 2')
                show_attenuation_plot_filter_2 = st.checkbox('Attenuation 2')

                st.markdown(f"<span style='color:{colour_material_3};'>Material 3</span>", unsafe_allow_html=True)
                show_transmission_plot_filter_3 = st.checkbox('Transmission 3')
                show_attenuation_plot_filter_3 = st.checkbox('Attenuation 3')

  
        # User input for filter materials
        energy_base_array, mass_atten_coeff_1, filter_1_material, filter_1_density, filter_1_thickness = select_attenuation(1,filter_material_selection_1,data_dir,modality,colour_material_1)
        
        # Determine a default value for the second filter that isn't the same as the first
        default_for_second_filter = filters[1] if filter_1_material == filters[0] else filters[0]
        energy_base_array_2, mass_atten_coeff_2, filter_2_material, filter_2_density, filter_2_thickness = select_attenuation(2,filter_material_selection_2,data_dir,modality,colour_material_2)

        # Determine a default value for the third filter that isn't the same as the first or second
        default_for_third_filter = filters[2] if filter_1_material == filters[0] else filters[0]
        energy_base_array_3, mass_atten_coeff_3, filter_3_material, filter_3_density, filter_3_thickness = select_attenuation(3,filter_material_selection_3,data_dir,modality,colour_material_3)

        with st.popover("Axis Options"):
        # Checkbox for turning grid on/off
            show_grid = st.checkbox("Show Grid", value=False)

            # Checkbox for scaling axes with selected kV
            scale_axes_with_kv = st.checkbox('Scale x-axis with selected kV')
            
            # Set the maximum tube voltage based selected kV or not for scaling the x-axis
            if scale_axes_with_kv:
                tube_voltage_max = tube_voltage
            
            y_axis_max = st.slider('Set maximum y-axis value:',min_value=0.001,max_value=1.0,step=0.001,value=1.0)

        with st.popover("Plot Style"):

            # Dropdown for selecting plot style
            selected_style = st.selectbox("Select Plot Style", plot_styles)

            # Dropdown for selecting plotting colour
            selected_colour = st.selectbox("Select Plot Colour", ["royalblue","tomato","teal","lightgreen","lightsteelblue","cyan","magenta","gold","darkorange","darkviolet","crimson","deeppink","grey"])


    with col2: # elements in col2 will be displayed in the right column
        col2a, col2b = st.columns(2) # create nested columns

        # Calculate the spectrum and get energy values below the tube voltage
        if mode: # Automatic mode
            energy_valid, energy_flux_normalised = kramers_law(target_material, energy_base_array, tube_voltage, tube_voltage_max, tube_voltage_min, current_time_product=current_time_product,current_time_product_max=current_time_product_max)

        else: # Manual mode
            energy_valid, energy_flux_normalised = kramers_law(target_material, energy_base_array, tube_voltage, tube_voltage_max, tube_voltage_min, tube_current, tube_current_max, exposure_time, exposure_time_max)

        # Calculate the filtered spectrum
        mass_atten_coeff_1_valid, relative_attenuation_filter_1 = relative_attenuation_mass_coeff(energy_base_array,filter_1_density,filter_1_thickness,mass_atten_coeff_1,tube_voltage,tube_voltage_min)
        mass_atten_coeff_2_valid, relative_attenuation_filter_2 = relative_attenuation_mass_coeff(energy_base_array,filter_2_density,filter_2_thickness,mass_atten_coeff_2,tube_voltage,tube_voltage_min)
        mass_atten_coeff_3_valid, relative_attenuation_filter_3 = relative_attenuation_mass_coeff(energy_base_array,filter_3_density,filter_3_thickness,mass_atten_coeff_3,tube_voltage,tube_voltage_min)

        # print(energy_flux_normalised.shape)
        # print(relative_attenuation_filter_1.shape)
        # print(relative_attenuation_filter_2.shape)
        # print(relative_attenuation_filter_3.shape)

        # Calculate the normalised energy flux filtered by all selected filters
        energy_flux_normalised_filtered = energy_flux_normalised * relative_attenuation_filter_1 * relative_attenuation_filter_2 * relative_attenuation_filter_3
        
        # Calculate the AUC percentage for the filtered spectrum
        auc_percentage = calculate_auc_percentage(energy_flux_normalised_filtered, energy_valid, 0, tube_voltage, tube_voltage_max)

        ###############################################################################################################################################################
        ########## Visualise the spectrum ########## 
        with st.container(border=True):
            # Create plotly figure
            fig = go.Figure()

            # Plot the spectrum with characteristic peaks or without
            fig.add_trace(go.Scatter(x=energy_valid, 
                                    y=energy_flux_normalised_filtered, 
                                    mode='lines', 
                                    line=dict(color=selected_colour, width=1.5), 
                                    name="Spectrum")
                         )

            # Add characteristic peaks
            if show_characteristic_xray_peaks:
                # Add characteristic peaks to the spectrum
                energy_valid, energy_flux_normalised_filtered, annotations = add_characteristic_peaks(target_material,energy_valid, energy_flux_normalised_filtered, tube_voltage)
                
                for ann in annotations:
                    if ann["energy"] <= tube_voltage:
                        peak_index = np.where(energy_valid == ann["energy"])[0]
                        if peak_index.size > 0:
                            peak_index = peak_index[0]
                            y_value = energy_flux_normalised_filtered[peak_index]
                            if y_value < ann["peak"]:
                                fig.add_annotation(x=ann["energy"], 
                                                y=y_value*0.95,
                                                text=ann["text"],
                                                showarrow=True,
                                                arrowhead=3,
                                                arrowsize=1,
                                                arrowwidth=1.2,
                                                ax=ann["xytext"][0],
                                                ay=ann["xytext"][1],
                                                font=dict(size=12))

            # Fill underneath the curve
            fig.add_trace(go.Scatter(x=energy_valid, 
                                    y=energy_flux_normalised_filtered, 
                                    mode='lines', 
                                    fill='tozeroy', 
                                    line=dict(color=selected_colour, width=1.5), 
                                    name="Filled Area")
                         )

            # Add vertical line for median energy
            if show_median_energy:
                # Call calculate_median_energy_and_hvl
                median_energy_at_50pct_auc = calculate_median_energy(energy_valid, energy_flux_normalised_filtered)
                
                median_index = np.where(energy_valid >= median_energy_at_50pct_auc)[0][0]
                median_height = energy_flux_normalised_filtered[median_index]
                
                # Add vertical line using add_shape
                fig.add_shape(
                    type="line",
                    x0=median_energy_at_50pct_auc,
                    x1=median_energy_at_50pct_auc,
                    y0=0,
                    y1=median_height,
                    line=dict(color="navy", width=0.8, dash="dash"),
                    # name=f"Median Energy: {median_energy_at_50pct_auc:.2f} keV"
                )
              
                fig.add_annotation(x=median_energy_at_50pct_auc, 
                                y=median_height / 2, 
                                text=f"E<sub>η</sub> = {median_energy_at_50pct_auc:.2f} keV", 
                                showarrow=False, 
                                font=dict(color="navy", size=14)
                )
            
            if show_mean_energy:
                # Calculate the mean energy using your function
                mean_energy = calculate_mean_energy(energy_valid, energy_flux_normalised_filtered)
                
                # Find the index corresponding to the mean energy
                mean_index = np.where(energy_valid >= mean_energy)[0][0]
                mean_height = energy_flux_normalised_filtered[mean_index]

                # Add a vertical line at the mean energy
                fig.add_shape(
                    type="line",
                    x0=mean_energy,
                    x1=mean_energy,
                    y0=0,
                    y1=mean_height,
                    line=dict(color="blueviolet", width=1.5, dash="dot"),
                )

                # Add annotation to show the mean energy value on the plot
                fig.add_annotation(x=mean_energy, 
                                y=mean_height / 8, 
                                text=f"E<sub>μ</sub> = {mean_energy:.2f} keV", 
                                showarrow=False, 
                                font=dict(color="blueviolet", size=14)
                )
            if show_peak_energy:
                # Calculate the peak energy using your function
                peak_energy = calculate_peak_energy(energy_valid, energy_flux_normalised_filtered)
                
                # Find the height of the peak energy flux
                peak_index = np.argmax(energy_flux_normalised_filtered)
                peak_height = energy_flux_normalised_filtered[peak_index]

                # Add a vertical line at the peak energy
                fig.add_shape(
                    type="line",
                    x0=peak_energy,
                    x1=peak_energy,
                    y0=0,
                    y1=peak_height,
                    line=dict(color="green", width=0.8, dash="dashdot"),
                )

                # Add annotation to show the peak energy value on the plot
                fig.add_annotation(x=peak_energy, 
                                y=peak_height * 1.1, 
                                text=f"E<sub>p</sub> = {peak_energy:.2f} keV", 
                                showarrow=False, 
                                font=dict(color="green", size=14)
                )

            # # Add vertical line for effective energy
            # if show_effective_energy:
                  # Call calculate_effective_energy_and_hvl
            #     energy_eff = calculate_effective_energy_and_hvl(energy_valid, energy_flux_normalised_filtered, filter_1_thickness)
        
            #     fig.add_vline(x=energy_eff, 
            #                 line=dict(color="darkgreen", width=0.8, dash="dash"), 
            #                 name=f"Effective Energy: {energy_eff:.2f} keV")
            #     fig.add_annotation(x=energy_eff, 
            #                     y=0.5, 
            #                     text=f"Effective Energy: {energy_eff:.2f} keV", 
            #                     showarrow=False, 
            #                     font=dict(color="darkgreen", size=8))

            # Add transmission and attenuation plots for filters
            if show_transmission_plot_filter_1:
                fig.add_trace(go.Scatter(x=energy_valid, 
                                        y=relative_attenuation_filter_1, 
                                        mode='lines', 
                                        line=dict(color=colour_material_1, width=1.5, dash="dash"), 
                                        name="Transmission Filter 1")
                             )

            if show_attenuation_plot_filter_1:
                fig.add_trace(go.Scatter(x=energy_valid, 
                                        y=mass_atten_coeff_1_valid, 
                                        mode='lines', 
                                        line=dict(color=colour_material_1a, width=2, dash="dot"), 
                                        name="Attenuation Filter 1", 
                                        yaxis="y2")
                             )

            if show_transmission_plot_filter_2:
                fig.add_trace(go.Scatter(x=energy_valid, 
                                        y=relative_attenuation_filter_2, 
                                        mode='lines', 
                                        line=dict(color=colour_material_2, width=1.5, dash="dash"), 
                                        name="Transmission Filter 2")
                             )

            if show_attenuation_plot_filter_2:
                fig.add_trace(go.Scatter(x=energy_valid, 
                                        y=mass_atten_coeff_2_valid, 
                                        mode='lines', 
                                        line=dict(color=colour_material_2a, width=2, dash="dot"), 
                                        name="Attenuation Filter 2", 
                                        yaxis="y2")
                             )

            if show_transmission_plot_filter_3:
                fig.add_trace(go.Scatter(x=energy_valid, 
                                        y=relative_attenuation_filter_3, 
                                        mode='lines', 
                                        line=dict(color=colour_material_3, width=2, dash="dash"), 
                                        name="Transmission Filter 3")
                             )

            if show_attenuation_plot_filter_3:
                fig.add_trace(go.Scatter(x=energy_valid, 
                                        y=mass_atten_coeff_3_valid, 
                                        mode='lines', 
                                        line=dict(color=colour_material_3a, width=1.5, dash="dot"), 
                                        name="Attenuation Filter 3", 
                                        yaxis="y2")
                             )

            # Annotate AUC percentage
            fig.add_annotation(
                x=0.95, 
                y=1.05, 
                text=f"AUC = {auc_percentage:.2f}%", 
                showarrow=False, 
                xref="paper", 
                yref="paper", 
                font=dict(color=selected_colour, size=30, family="sans-serif")
            )

             # Check if user changed the y-axis range (e.g., via zoom or autoscale)
            if "last_y_range" not in st.session_state:
                st.session_state.last_y_range = fig.layout.yaxis.range  # Store the initial hardcoded range
        
            # Preserve user changes to the y-axis range
            if fig.layout.yaxis.range != st.session_state.last_y_range:
                st.session_state.last_y_range = fig.layout.yaxis.range  # Update stored range if user changed it
          

            # Update layout
            fig.update_layout(
                xaxis=dict(
                  title="Photon Energy E (keV)",
                  range=[0, tube_voltage_max],
                  dtick=10, 
                  showgrid=False,
                  size=20,
                ),
                yaxis=dict(
                  title="Relative Energy Flux Φ",
                  range=[0, y_axis_max],
                  dtick=0.1, 
                  showgrid=False,
                  size=20,
                ),
                yaxis2=dict(
                    title="Mass Attenuation Coefficient μ (cm²/g)",
                    overlaying='y',
                    side='right',
                    type='log',
                    showgrid=False,
                    size=20,
                ),
                showlegend=False,
                template=selected_style,
                width=1300,   # Set the width of the figure, also limited by col width
                height=720,    # Set the height of the figure
                uirevision='constant', # Don't reset axes on every user input change
            )

            # Add grid if needed
            if show_grid:
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=True)

            # Render plot with Streamlit
            st.plotly_chart(fig)
