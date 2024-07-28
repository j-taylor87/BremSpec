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

            "filters": ["Al (Z=13)", "Cu (Z=29)","PMMA (Z~7)"],

            "automatic_mode": "Automatic Exposure Control (AEC) (WIP)"
        }

    elif modality == "Mammography":
        return {
            "tube_voltage_max": 50.0,
            "tube_voltage_min": 10.0,
            "tube_voltage_default": 30.0,

            "tube_current_max": 100.0,
            "tube_current_min": 1.0,
            "tube_current_default": 80.0,

            "exposure_time_max": 200.0,
            "exposure_time_min": 1.0,
            "exposure_time_default": 150.0,

            "current_time_product_max": 100.0,
            "current_time_product_min": 1.0,
            "current_time_product_default": 20.0,

            "filters": ["Al (Z=13)", "Mo (Z=42)", "Rh (Z=45)", "Ag (Z=47)"],

            "automatic_mode": "Automatic Exposure Control (AEC) (WIP)"
        }

    elif modality == "Fluoroscopy (WIP)":
        return {
            "tube_voltage_max": 133.0,
            "tube_voltage_min": 40.0,
            "tube_voltage_default": 50.0,

            "tube_current_max": 500.0,
            "tube_current_min": 1.0,
            "tube_current_default": 100.0,

            "exposure_time_max": 1000.0,
            "exposure_time_min": 1.0,
            "exposure_time_default": 0.1,

            "pulse_width_max": 20.0,
            "pulse_width_min": 1.0,
            "pulse_width_default": 8.0,

            "filters": ["Al (Z=13)", "Cu (Z=29)"],

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

            "exposure_time_max": 2.0,
            "exposure_time_min": 0.0,
            "exposure_time_default": 0.5,

            "filters": ["Al (Z=13)", "Cu (Z=29)", "Sn (Z=50)"],

            "automatic_mode": "Automatic Exposure Control (AEC) (WIP)"
        }

    else:
        return {}
