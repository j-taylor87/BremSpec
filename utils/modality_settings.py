# modality_settings.py
# Project: BremSpec
# Author: James Taylor
# Date: October 2023
# Modality settings utility for different imaging modalities
import streamlit as st

@st.fragment
def get_modality_settings(modality):
    if modality == "General X-ray":
        return {
            "tube_voltage_max": 150.0,
            "tube_voltage_min": 40.0,
            "tube_voltage_default": 125.0,

            "tube_current_max": 500.0,
            "tube_current_min": 1.0,
            "tube_current_default": 400.0,

            "exposure_time_max": 1000.0,
            "exposure_time_min": 1.0,
            "exposure_time_default": 800.0,

            "current_time_product_max": 500.0,
            "current_time_product_min": 0.0,
            "current_time_product_default": 100.0,

            # area at detector, cm²
            "field_area_cm2_min": 100.0,   # ~10×10
            "field_area_cm2_max": 1600.0,  # ~40×40
            "field_area_cm2_default": 1600.0, # ~20×20

            # NEW: Source–Image Distance (cm)
            "sid_cm_min": 80.0,
            "sid_cm_max": 150.0,
            "sid_cm_default": 110.0,

            "filters": ["Al (Z=13)", 
                        "Cu (Z=29)",
                        "PMMA (Zeff~6.56)",
                        "Soft Tissue (Zeff~7.52)",
                        "Cortical Bone (Zeff~13.98)",
                        "Lung Tissue (Zeff~8.0)",
                        "Adipose Tissue (Zeff~6.44)",
                        "CaSO4 (Gypsum) (Zeff~13)",
                        "Pb (Z=82)",
            ],

            "automatic_mode": "Automatic mAs (WIP)"
        }

    elif modality == "Mammography":
        return {
            "tube_voltage_max": 40.0,
            "tube_voltage_min": 10.0,
            "tube_voltage_default": 26.0,

            "tube_current_max": 150.0,
            "tube_current_min": 1.0,
            "tube_current_default": 120.0,

            "exposure_time_max": 2000.0,
            "exposure_time_min": 1.0,
            "exposure_time_default": 1000.0,

            "current_time_product_max": 100.0,
            "current_time_product_min": 1.0,
            "current_time_product_default": 20.0,

            # (paddle sizes ~18×24=432, 24×30=720
            "field_area_cm2_min": 200.0,
            "field_area_cm2_max": 720.0,
            "field_area_cm2_default": 720.0,

            "filters": ["Rh (Z=45)",
                        "Mo (Z=42)",
                        "Ag (Z=47)",
                        "Be (Z=4)",
                        "Al (Z=13)", 
                        "I (Z=53)",
                        "PMMA (Zeff~6.56)",
                        "Breast Tissue (Zeff~7.88)",
                        "Adipose Tissue (Zeff~6.44)",
                        "Ca (Z=20)",
                        "CaSO4 (Gypsum) (Zeff~13)"
            ],

            "automatic_mode": "Automatic mAs (WIP)"
        }

    elif modality == "Fluoroscopy":
        return {
            "tube_voltage_max": 133.0,
            "tube_voltage_min": 40.0,
            "tube_voltage_default": 120.0,

            "tube_current_max": 500.0,
            "tube_current_min": 1.0,
            "tube_current_default": 400.0,

            "pulse_width_max": 100.0,
            "pulse_width_min": 1.0,
            "pulse_width_default": 10.0,

            # typical FOV/collected area
            "field_area_cm2_min": 50.0,
            "field_area_cm2_max": 1250.0,
            "field_area_cm2_default": 1250.0,

            "filters": ["Al (Z=13)", 
                        "Cu (Z=29)",
                        "I (Z=53)",
                        "PMMA (Zeff~6.56)",
                        "Soft Tissue (Zeff~7.52)",
                        "Cortical Bone (Zeff~13.98)",
                        "Lung Tissue (Zeff~8.0)",
                        "Adipose Tissue (Zeff~6.44)",
                        "CaSO4 (Gypsum) (Zeff~13)",
                        "Pb (Z=82)"
            ],

            "automatic_mode": "Automatic Exposure Rate Control (AERC) (WIP)"
        }

    elif modality == "CT":
        return {
            "tube_voltage_max": 140.0,
            "tube_voltage_min": 50.0,
            "tube_voltage_default": 120.0,

            "tube_current_max": 1000.0,
            "tube_current_min": 0.0,
            "tube_current_default": 500.0,

            # Rotation time — keep ms for manual paths (existing code)
            "exposure_time_max": 2000.0,   # ms
            "exposure_time_min": 100.0,    # ms (avoid 0)
            "exposure_time_default": 500.0,# ms

            # Optional: seconds for CT auto UI (use if you wire it in)
            "rotation_time_max_s": 2.0,
            "rotation_time_min_s": 0.3,
            "rotation_time_default_s": 0.5,

            # NEW: scanner geometry / technique controls
            "pitch_max": 2.0,
            "pitch_min": 0.5,
            "pitch_default": 1.0,

            "collimation_max_mm": 160.0,    # total collimation (detector width) along table
            "collimation_min_mm": 5.0,
            "collimation_default_mm": 80.0,

            "filters": [
                "Al (Z=13)",
                "Cu (Z=29)",
                "Sn (Z=50)",
                "I (Z=53)",
                "PMMA (Zeff~6.56)",
                "Soft Tissue (Zeff~7.52)",
                "Cortical Bone (Zeff~13.98)",
                "Lung Tissue (Zeff~8.0)",
                "Adipose Tissue (Zeff~6.44)",
                "CaSO4 (Gypsum) (Zeff~13)",
                "Pb (Z=82)",
            ],

            "automatic_mode": "Dose Modulation (WIP)",
        }


    else:
        return {}
