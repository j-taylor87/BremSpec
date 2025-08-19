# app.py
# Project: BremSpec
# Author: James Taylor
# Date: October 2023

import streamlit as st
# print("Streamlit version:", st.__version__)

# Import UI components
from ui.ui_options_and_styles import MODALITIES, PLOT_STYLES
from ui.panel_left import render_panel_left
from ui.panel_right import render_panel_right
from ui.panel_centre import render_panel_centre

# --- Fragments: only these parts re-run when their widgets change ---

@st.fragment
def left_and_plot(left_area, plot_area, modalities, PLOT_STYLES):
    with left_area.container():
        left = render_panel_left(modalities=MODALITIES)
    st.session_state["left"] = left  # share with other fragments

    # Only draw if right panel has run at least once
    right = st.session_state.get("right")
    if right:
        with plot_area.container():
            render_panel_centre(left, right, PLOT_STYLES)


@st.fragment
def right_and_plot(right_area, plot_area, data_dir, PLOT_STYLES):
    left = st.session_state.get("left")
    # If left hasn't run yet, we can't render right (needs filters/modality)
    with right_area.container():
        if not left:
            st.info("Select modality and technique first.")
            return
        right = render_panel_right(filters=left["filters"], data_dir=data_dir, modality=left["modality"])
    st.session_state["right"] = right

    with plot_area.container():
        render_panel_centre(left, right, PLOT_STYLES)

# Set data directory
data_dir = "./data" # Works with GitHub

# Set streamlit page to wide mode
st.set_page_config(
    layout="wide", 
    page_icon='☢️',
    page_title="BremSpec",
)

PANEL_H = 860

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

@st.fragment
def render_main():
    st.title("BremSpec")
    st.markdown("<h4 style='color: #666;'>Bremsstrahlung X-ray Spectrum Visualiser</h2>", unsafe_allow_html=True)

    # All layout/containers are created inside the fragment
    col1, col2, col3 = st.columns([0.8, 2.2, 0.6])

    with col1:
        with st.container(
            # height=PANEL_H,
            border=True, 
            key="left-panel"
            ):
            left = render_panel_left(modalities=MODALITIES)

    with col3:
        with st.container(
            # height=PANEL_H, 
            border=True, 
            key="right-panel"):
            right = render_panel_right(filters=left["filters"], data_dir=data_dir, modality=left["modality"])

    with col2:
        with st.container(
            # height=PANEL_H, 
            border=True, 
            key="centre-panel"):
            render_panel_centre(left, right, PLOT_STYLES)

def app():
    render_main()

if __name__ == "__main__":
    app()