# app.py
# Project: BremSpec
# Author: James Taylor
# Date: October 2023

import streamlit as st
# print("Streamlit version:", st.__version__)
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

# Custom functions
from utils.modality_settings import get_modality_settings
from utils.select_attenuation_and_load_data import select_attenuation
from utils.calc_utils import (
    add_characteristic_peaks,
    calculate_auc_percentage,
    calculate_effective_energy_and_hvl,
    calculate_mean_energy,
    calculate_median_energy,
    calculate_peak_energy,
    kramers_law,
    relative_attenuation_mass_coeff
)
from utils.calc_utils import calculate_compton_scatter_spectrum
from ui.ui_options import plot_styles, modalities

# Set data directory
data_dir = "./data" # Works with GitHub

# Set streamlit page to wide mode
st.set_page_config(
    layout="wide", 
    page_icon='☢️',
    page_title="BremSpec",
)

# Current (2024) CSS workaround for edited whitespace of app
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
    st.markdown("<h4 style='color: #666;'>Bremsstrahlung X-ray Spectrum Visualiser</h2>", unsafe_allow_html=True) #`Decelerate and Illuminate :P`
    
    # Create columns and specify widths
    col1, col2, col3 = st.columns([0.8,2.2,0.6])

    with col1: # elements in col1will display in the left column
        with st.container(border=True):
            
            # User input for modality
            modality = st.selectbox("Modality", modalities)  # Add more modalities as needed

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

            # User input for mode
            mode = st.toggle(automatic_mode,value=False)

            if "tube_voltage_old" not in st.session_state:
                st.session_state.tube_voltage_old = tube_voltage_default  # Default tube voltage

            if "current_time_product_old" not in st.session_state:
                st.session_state.current_time_product_old = tube_current_default*exposure_time_default / 1000.0  # Default current-time product
            
            # User input for technique factors based on selected mode
            if mode: # Automatic mode
                tube_voltage = st.slider("Tube Voltage (kV)", min_value=int(tube_voltage_min), max_value=int(tube_voltage_max), value=int(tube_voltage_default))
                
                if modality == "CT":
                    tube_current = 1/tube_voltage**5.0
                    exposure_time = st.slider("Rotation Time (s)", min_value=exposure_time_min, max_value=exposure_time_max, value=exposure_time_default,format="%.2f")
                    
                    # Calculate the new current-time product
                    current_time_product = st.session_state.current_time_product_old*(st.session_state.tube_voltage_old / tube_voltage) ** 2.0
                    current_time_product_display = st.write("Current-Time Product (mAs): ", round(current_time_product,0))
                    
                    # Update the old values for the next run
                    st.session_state.tube_voltage_old = tube_voltage
                    st.session_state.current_time_product_old = current_time_product
                    
                elif modality == "Mammography":
                    
                    # Calculate the new current-time product
                    current_time_product = st.session_state.current_time_product_old*(st.session_state.tube_voltage_old / tube_voltage) ** 2.0
                    current_time_product_display = st.write("Current-Time Product (mAs): ", round(current_time_product,0))
                    
                    # Update the old values for the next run
                    st.session_state.tube_voltage_old = tube_voltage
                    st.session_state.current_time_product_old = current_time_product

                else: # e.g. General X-ray

                    # Calculate the new current-time product
                    current_time_product = st.session_state.current_time_product_old*(st.session_state.tube_voltage_old / tube_voltage) ** 2.0
                    current_time_product_display = st.write("Current-Time Product (mAs): ", round(current_time_product,0))
                    
                    # Update the old values for the next run
                    st.session_state.tube_voltage_old = tube_voltage
                    st.session_state.current_time_product_old = current_time_product

            else: # Manual mode
                tube_voltage = st.slider("Tube Voltage (kV)", min_value=int(tube_voltage_min), max_value=int(tube_voltage_max), value=int(tube_voltage_default))
                tube_current = st.slider("Tube Current (mA)", min_value=int(tube_current_min), max_value=int(tube_current_max), value=int(tube_current_default))
                
                if modality == "CT":
                    exposure_time = st.slider("Rotation Time (ms)", min_value=exposure_time_min, max_value=exposure_time_max, value=exposure_time_default,format="%.0f")
                elif modality == "Fluoroscopy":
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

            # Checkbox for showing charactersistic X-ray peaks
            show_characteristic_xray_peaks = st.checkbox("Characteristic X-ray Peaks", value=False)

            # Checkbox for showing the effective beam energy
            show_effective_energy = st.checkbox("Effective Energy Eeff and HVL", value=False)

            # Checkbox for showing the median beam energy
            show_median_energy = st.checkbox("Median Energy Eη", value=False)

            # Checkbox for showing the mean beam energy
            show_mean_energy = st.checkbox("Mean Energy Eμ", value=False)

            # Checkbox for showing the mean beam energy
            show_peak_energy = st.checkbox("Peak Energy Ep", value=False)

            with st.popover("Instructions",use_container_width=True):
                st.markdown(""" - Select an X-ray imaging modality, technique factors (kV, mA, ms), and filter/target materials to see how these factors affect the 
                        shape of the Bremsstrahlung X-ray spectrum.
                        \n - The available energies along the x-axis will depend on the selected modality, and the x-axis can be set to scale with kV.
                        \n - The y-axis represents the relative energy flux: the energy flux of each energy normalised to the maximum energy flux.
                        \n - Scaling, zooming, and panning tools can be found on the top right of the graph when hovering over it with the mouse. "Axis Options" contains related settings.
                        \n - The area under the curve (AUC), representing the normalised total beam energy across all available 
                        energies, is displayed in the top-right corner of the plot. This is a relative percentage of the unfiltered beam at maximum technique factors of the selected modality, with a tungsten target. 
                        If scaling x-axis with selected kV is selected, then the AUC calculation will take this as the maximum kV.
                        \n - Effective beam energy can be displayed as a vertical line on the graph. HVL is also displayed, which is based on Material 1 attenuation.
                        \n - Median, mean, and peak beam energy can also be displayed as vertical lines on the graph for comparison.
                        \n - The characteristic X-rays of the anode target material can be displayed.
                        \n - Attenuation and tranmission plots can be viewed for the different filter materials. This is especially useful for visualising absorption edges.
                        \n - Attenuation and characteristic X-ray data obtained from NIST (2004) https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients"""
                )

    with col3: # col3 before col2 to define the show grid button    
        with st.container(border=True):
            # User input for target material
            target_material = st.selectbox("Anode Target Material", ["W (Z=74)", "Rh (Z=45)", "Mo (Z=42)"])

            # Default values for each filter
            default_filter_1 = filters[0]
            default_filter_2 = filters[1]
            default_filter_3 = filters[2]
            default_attenuation_4 = filters[2]

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
                with st.popover("Attenuator Materials",use_container_width=True):
                    # Selection boxes for filters
                    filter_material_selection_1 = st.selectbox(f"Material 1", filters, index=default_index_1, key=f"filter_material_1")
                    filter_material_selection_2 = st.selectbox(f"Material 2", filters, index=default_index_2, key=f"filter_material_2")
                    filter_material_selection_3 = st.selectbox(f"Material 3", filters, index=default_index_3, key=f"filter_material_3")

            with st.container():
                with st.popover("Attenuation/Transmission Overlays",use_container_width=True):
                    # Display the text in color above each checkbox
                    st.markdown(f"<span style='color:{colour_material_1};'>Material 1</span>", unsafe_allow_html=True)
                    show_attenuation_plot_filter_1 = st.checkbox('Attenuation 1')
                    show_transmission_plot_filter_1 = st.checkbox('Transmission 1')
                              
                    st.markdown(f"<span style='color:{colour_material_2};'>Material 2</span>", unsafe_allow_html=True)
                    show_attenuation_plot_filter_2 = st.checkbox('Attenuation 2')
                    show_transmission_plot_filter_2 = st.checkbox('Transmission 2')
                    
                    st.markdown(f"<span style='color:{colour_material_3};'>Material 3</span>", unsafe_allow_html=True)
                    show_attenuation_plot_filter_3 = st.checkbox('Attenuation 3')
                    show_transmission_plot_filter_3 = st.checkbox('Transmission 3')
                    

            # User input for filter materials
            energy_base_array, mass_atten_coeff_1, filter_1_material, filter_1_density, filter_1_thickness = select_attenuation(1,filter_material_selection_1,data_dir,modality,colour_material_1)
            
            # Determine a default value for the second filter that isn't the same as the first
            default_for_second_filter = filters[1] if filter_1_material == filters[0] else filters[0]
            energy_base_array_2, mass_atten_coeff_2, filter_2_material, filter_2_density, filter_2_thickness = select_attenuation(2,filter_material_selection_2,data_dir,modality,colour_material_2)

            # Determine a default value for the third filter that isn't the same as the first or second
            default_for_third_filter = filters[2] if filter_1_material == filters[0] else filters[0]
            energy_base_array_3, mass_atten_coeff_3, filter_3_material, filter_3_density, filter_3_thickness = select_attenuation(3,filter_material_selection_3,data_dir,modality,colour_material_3)

            with st.popover("Scatter Overlay",use_container_width=True):
                show_scatter_plot = st.toggle("Show Scatter Spectrum", value=False)
                scatter_angle_deg = st.slider("Compton Scatter Angle (degrees)", min_value=0, max_value=360, value=90, step=1)
                scatter_material = st.selectbox("Scatter Attenuator Material", filters, key="scatter_material")
                show_scatter_eff_hvl = st.checkbox("Show Effective Energy & HVL (scatter)", value=False)
                scatter_energy_base, scatter_mass_atten, scatter_material_name, scatter_density, scatter_thickness = select_attenuation(
                    4, scatter_material, data_dir, modality, 'deepskyblue'
                )
                scatter_y_scale = st.slider(
                    "Scatter Intensity Scaling (% of primary maximum)",
                    min_value=0.01, max_value=100.0, value=10.0, step=0.01,
                    help="0.1% of primary typical for lateral scatter at 1 m"
                )

    with col2:
        with st.container(border=True):

            y_axis_max = st.session_state.get("y_axis_max", 1.0)
            scale_axes_with_kv = st.session_state.get("scale_axes_with_kv", False)
            show_grid = st.session_state.get("show_grid", False)
            selected_style = st.session_state.get("selected_style", plot_styles[0])
            selected_colour = st.session_state.get("selected_colour", "royalblue")

            

            # Calculate the spectrum and get energy values below the tube voltage
            if mode: # Automatic mode
                energy_valid, energy_flux_normalised = kramers_law(target_material, energy_base_array, tube_voltage, tube_voltage_max, tube_voltage_min, current_time_product=current_time_product,current_time_product_max=current_time_product_max)

            else: # Manual mode
                energy_valid, energy_flux_normalised = kramers_law(target_material, energy_base_array, tube_voltage, tube_voltage_max, tube_voltage_min, tube_current, tube_current_max, exposure_time, exposure_time_max)

            # Calculate the filtered spectrum
            mass_atten_coeff_1_valid, relative_attenuation_filter_1 = relative_attenuation_mass_coeff(energy_base_array,filter_1_density,filter_1_thickness,mass_atten_coeff_1,tube_voltage)
            mass_atten_coeff_2_valid, relative_attenuation_filter_2 = relative_attenuation_mass_coeff(energy_base_array,filter_2_density,filter_2_thickness,mass_atten_coeff_2,tube_voltage)
            mass_atten_coeff_3_valid, relative_attenuation_filter_3 = relative_attenuation_mass_coeff(energy_base_array,filter_3_density,filter_3_thickness,mass_atten_coeff_3,tube_voltage)

            # Calculate the normalised energy flux filtered by all selected filters
            energy_flux_normalised_filtered = energy_flux_normalised * relative_attenuation_filter_1 * relative_attenuation_filter_2 * relative_attenuation_filter_3
            
            # Calculate the AUC percentage for the filtered spectrum
            auc_percentage = calculate_auc_percentage(energy_flux_normalised_filtered, energy_valid, 0, tube_voltage, tube_voltage_max)

            ########## Visualise the spectrum ########## 
                
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
                                                font=dict(size=16))

            

            # Convert the selected_colour to RGBA for the fill, with desired opacity
            rgb_tuple = mcolors.to_rgb(selected_colour) # Get RGB values (0-1 range)
            r, g, b = [int(255 * c) for c in rgb_tuple] # Convert to 0-255 range
            fill_color_rgba = f"rgba({r},{g},{b},0.2)" # Create RGBA string with 0.2 opacity

            # Fill underneath the curve
            fig.add_trace(go.Scatter(x=energy_valid, 
                                    y=energy_flux_normalised_filtered, 
                                    mode='lines',  
                                    fill='tozeroy', 
                                    line=dict(color=selected_colour, width=1.5),
                                    fillcolor=fill_color_rgba,
                                    name="Filled Area")
                        )
            
            # Add vertical line for effective energy
            if show_effective_energy:

                effective_energy, t_hvl = calculate_effective_energy_and_hvl(
                    energy_valid,                           
                    energy_flux_normalised_filtered,                           
                    mass_atten_coeff_1_valid,                 
                    filter_1_density                                 
                )

                effective_energy_index = np.where(energy_valid >= effective_energy)[0][0]
                effective_energy_height = energy_flux_normalised_filtered[effective_energy_index]

                # Add vertical line using add_shape
                fig.add_shape(
                    type="line",
                    x0=effective_energy,
                    x1=effective_energy,
                    y0=0,
                    y1=effective_energy_height,
                    line=dict(color="blue", width=2, dash="dash"),
                )

                fig.add_annotation(
                    x=effective_energy,
                    y=effective_energy_height * 1.25,  # choose appropriate y for your plot
                        text=(
                        f"E<sub>eff</sub> = {effective_energy:.2f} keV"
                        "<br>"
                        f"HVL = {t_hvl*10:.2f} mm {filter_1_material}"
                    ),
                    showarrow=False,
                    font=dict(color="blue", size=18)
                )

            # Add vertical line for median energy
            if show_median_energy:
                median_energy_at_50pct_auc = calculate_median_energy(
                    energy_valid, 
                    energy_flux_normalised_filtered
                )
                
                median_index = np.where(energy_valid >= median_energy_at_50pct_auc)[0][0]
                median_height = energy_flux_normalised_filtered[median_index]
                
                # Add vertical line using add_shape
                fig.add_shape(
                    type="line",
                    x0=median_energy_at_50pct_auc,
                    x1=median_energy_at_50pct_auc,
                    y0=0,
                    y1=median_height,
                    line=dict(color="green", width=2, dash="dash"),
                    # name=f"Median Energy: {median_energy_at_50pct_auc:.2f} keV"
                )
            
                fig.add_annotation(x=median_energy_at_50pct_auc, 
                                y=median_height / 2, 
                                text=f"E<sub>η</sub> = {median_energy_at_50pct_auc:.2f} keV", 
                                showarrow=False, 
                                font=dict(color="green", size=16)
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
                    line=dict(color="blueviolet", width=3, dash="dot"),
                )

                # Add annotation to show the mean energy value on the plot
                fig.add_annotation(x=mean_energy, 
                                y=mean_height / 8, 
                                text=f"E<sub>μ</sub> = {mean_energy:.2f} keV", 
                                showarrow=False, 
                                font=dict(color="blueviolet", size=16)
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
                    line=dict(color="darkorange", width=3, dash="dashdot"),
                )

                # Add annotation to show the peak energy value on the plot
                fig.add_annotation(x=peak_energy, 
                                y=peak_height * 1.05, 
                                text=f"E<sub>p</sub> = {peak_energy:.2f} keV", 
                                showarrow=False, 
                                font=dict(color="orange", size=16)
                )

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
                text=f"Total Energy = {auc_percentage:.2f}%", 
                showarrow=False, 
                xref="paper", 
                yref="paper", 
                font=dict(color=selected_colour, size=25, family="sans-serif")
            )
        
            # Update layout
            fig.update_layout(
                xaxis=dict(
                title="Photon Energy E (keV)",
                range=[0, tube_voltage_max],
                dtick=10, 
                showline=True,           # show axis line
                linewidth=3,             # thickness of axis line
                showgrid=False,
                title_font=dict(size=20),      # Axis title font size
                tickfont=dict(size=18),        # Tick label font size
                ),
                yaxis=dict(
                title="Relative Energy Flux Ψ",
                range=[0, y_axis_max],
                dtick=0.1, 
                showline=True,           # show axis line
                linewidth=3,             # thickness of axis line
                showgrid=False,
                title_font=dict(size=22),
                tickfont=dict(size=18),
                ),
                yaxis2=dict(
                    title="Mass Attenuation Coefficient μ (cm²/g)",
                    overlaying='y',
                    side='right',
                    type='log',
                    showgrid=False,
                title_font=dict(size=22),
                    tickfont=dict(size=18),
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


            # Scatter plot
            if show_scatter_plot:

                # Simple mapping of primary to contrast
                contrast_map = {
                    "royalblue": "orange",
                    "deepskyblue": "darkorange",
                    "tomato": "blue",
                    "magenta": "lime",
                    "cyan": "crimson",
                    "lightgreen": "darkviolet",
                    # etc. Add as needed.
                }
                scatter_color = contrast_map.get(selected_colour, "orange")  # default to orange

                scatter_energy, scatter_flux = calculate_compton_scatter_spectrum(
                    energy_valid,
                    energy_flux_normalised_filtered,
                    scatter_angle_deg,
                    scatter_mass_atten,
                    scatter_density,
                    scatter_thickness,
                    scatter_energy_base
                )

                primary_max = np.max(energy_flux_normalised_filtered)
                physical_scatter_scale = 0.001  # 0.1% at 1m

                # Final scale = physical * user slider
                scatter_flux_scaled = scatter_flux * (primary_max * (scatter_y_scale / 100.0))

                fig.add_trace(go.Scatter(
                    x=scatter_energy,
                    y=scatter_flux_scaled,
                    mode='lines',
                    line=dict(color="orange", width=2, dash="dot"),
                    name=f"Compton Scatter @ {scatter_angle_deg}°"
                ))

                if show_scatter_eff_hvl:
                    # Interpolate attenuation to energy grid
                    if len(scatter_mass_atten) != len(scatter_energy):
                        mass_atten_interp = interp1d(scatter_energy_base, scatter_mass_atten, bounds_error=False, fill_value="extrapolate")
                        scatter_mass_atten_interp = mass_atten_interp(scatter_energy)
                    else:
                        scatter_mass_atten_interp = scatter_mass_atten

                    eff_energy_scatter, hvl_scatter = calculate_effective_energy_and_hvl(
                        scatter_energy, scatter_flux_scaled, scatter_mass_atten_interp, scatter_density
                    )
                    fig.add_shape(
                        type="line",
                        x0=eff_energy_scatter, x1=eff_energy_scatter,
                        y0=0, y1=np.interp(eff_energy_scatter, scatter_energy, scatter_flux_scaled),
                        line=dict(color="orange", width=4, dash="dash"),
                    )
                    fig.add_annotation(
                        x=eff_energy_scatter,
                        y=np.interp(eff_energy_scatter, scatter_energy, scatter_flux_scaled),
                        text=(
                            f"E<sub>eff</sub> (scatter) = {eff_energy_scatter:.2f} keV"
                            "<br>"
                            f"HVL = {hvl_scatter*10:.2f} mm {scatter_material}"
                        ),
                        showarrow=False,
                        xshift=160,
                        yshift=20,
                        font=dict(color="orange", size=16)
                    )


            # Render plot with Streamlit
            st.plotly_chart(fig,use_container_width=True)

            with st.container():
                controls = st.columns([1, 1, 1, 2])
                with controls[0]:
                    y_axis_max = st.slider(
                        "Y Axis Max",
                        min_value=0.01, max_value=1.0, value=y_axis_max, step=0.01,
                        help="Set the maximum value of the Y axis",
                        key="y_axis_max"
                    )
                with controls[1]:
                    scale_axes_with_kv = st.checkbox(
                        'Scale x-axis with selected kV', value=scale_axes_with_kv, key="scale_axes_with_kv"
                    )
                with controls[2]:
                    show_grid = st.checkbox(
                        "Show Grid", value=show_grid, key="show_grid"
                    )
                with controls[3]:
                    with st.popover("Plot Style"):
                        selected_style = st.selectbox(
                            "Select Plot Style",
                            plot_styles,
                            index=plot_styles.index(selected_style) if selected_style in plot_styles else 0,
                            key="selected_style"
                        )
                        selected_colour = st.selectbox(
                            "Select Plot Colour",
                            ["royalblue", "orange", "lime", "magenta", "cyan", "gold", "crimson", "deeppink", "grey"],
                            index=0,
                            key="selected_colour"
                        )

