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

    elif modality == "Mammography (WIP)":
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

            "filters": ["Be (Z=4)",
                        "Al (Z=13)", 
                        "Mo (Z=42)", 
                        "Rh (Z=45)", 
                        "Ag (Z=47)",
                        "I (Z=53)",
                        "PMMA (Zeff~6.56)",
                        "Breast Tissue (Zeff~7.88)",
                        "Adipose Tissue (Zeff~6.44)",
                        "Ca (Z=20)",
                        "CaSO4 (Gypsum) (Zeff~13)"
            ],

            "automatic_mode": "Automatic mAs (WIP)"
        }

    elif modality == "Fluoroscopy (WIP)":
        return {
            "tube_voltage_max": 133.0,
            "tube_voltage_min": 40.0,
            "tube_voltage_default": 50.0,

            "tube_current_max": 500.0,
            "tube_current_min": 1.0,
            "tube_current_default": 200.0,

            "exposure_time_max": 100.0,
            "exposure_time_min": 1.0,
            "exposure_time_default": 10.0,

            "pulse_width_max": 20.0,
            "pulse_width_min": 1.0,
            "pulse_width_default": 8.0,

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

            "automatic_mode": "Automatic Dose Rate Control (ADRC) (WIP)"
        }

    elif modality == "CT (WIP)":
        return {
            "tube_voltage_max": 140.0,
            "tube_voltage_min": 50.0,
            "tube_voltage_default": 120.0,

            "tube_current_max": 1000.0,
            "tube_current_min": 0.0,
            "tube_current_default": 500.0,

            "exposure_time_max": 2000.0,
            "exposure_time_min": 0.0,
            "exposure_time_default": 500.0,

            "filters": ["Al (Z=13)", 
                        "Cu (Z=29)", 
                        "Sn (Z=50)", 
                        "I (Z=53)",
                        "I (Z=53)",
                        "PMMA (Zeff~6.56)",
                        "Soft Tissue (Zeff~7.52)",
                        "Cortical Bone (Zeff~13.98)",
                        "Lung Tissue (Zeff~8.0)",
                        "Adipose Tissue (Zeff~6.44)",
                        "CaSO4 (Gypsum) (Zeff~13)",
                        "Pb (Z=82)",
            ],

            "automatic_mode": "Dose Modulation (WIP)"
        }

    else:
        return {}
