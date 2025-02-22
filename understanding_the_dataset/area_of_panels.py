import pandas as pd
import matplotlib.pyplot as plt
import os
"""
Method used to compute the area (in meters) for a single instance:
From the geotransform given in the readme:

1. Each label file contains the width and height (normalized) of each solar panel.
2. The chip size for native images is 416 x 416 pixels.
3. The resolution of the imagery is 31 cm (0.31 meters) per pixel.
4. The formula to compute the area of each solar panel in meters is:

Area = (Width * 416 * 0.31) * (Height * 416 * 0.31)
"""

native_labels_path = 'C://Users//aarav//Downloads//labels-20250212T103318Z-001//labels//labels_native'

widths = []
heights = []
for root, dirs, files in os.walk(native_labels_path):
    for filename in files:
        full_path = os.path.join(root, filename)
        with open(full_path, 'r') as fp:
            for line in fp:
                width = float(line.strip().split()[3]) # width and height are on the 3rd and 4th indices
                height = float(line.strip().split()[4])
                widths.append(width)
                heights.append(height)

chip_size = 416  
pixel_resolution = 0.31 

df = pd.DataFrame({'width':widths, 'height': heights})
df["area_m2"] = (df["width"] * chip_size * pixel_resolution) * (df["height"] * chip_size * pixel_resolution)

mean_area = df["area_m2"].mean()
std_area = df["area_m2"].std()

print(f"Mean area: {mean_area:.2f} m² \n")
print(f"Standard deviation of area: {std_area:.2f} m²")

plt.hist(df["area_m2"], bins=10, range = (0, 1000))
plt.xlabel("Area (m^2)")
plt.ylabel("Frequency")
plt.title("Histogram of Solar Panel Areas")
plt.show()

'''
Observations:
Most solar panels have a small area.
Frequency: A majority of the solar panels have areas below 200 m².
Outliers: There are a few panels with large areas (600-1000 m²), but they are rare.
'''