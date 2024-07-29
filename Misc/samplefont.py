import matplotlib.pyplot as plt
from matplotlib import font_manager

# List all available font families
available_fonts = set(f.name for f in font_manager.fontManager.ttflist)

# Check if the desired font family is available
desired_font = "Times New Roman"  # Change to your desired font family
if desired_font in available_fonts:
    plt.rcParams["font.family"] = desired_font
else:
    print(f"Font '{desired_font}' not found. Available fonts: {available_fonts}")

# Example plot to test the font setting
plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
plt.title("Sample Plot with Custom Font")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()