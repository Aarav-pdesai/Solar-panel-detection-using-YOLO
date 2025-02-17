import pandas as pd
import os
import matplotlib.pyplot as plt
native_labels_path = 'C://Users//aarav//Downloads//labels-20250212T103318Z-001//labels//labels_native'

PV_modules_in_each_image = []
for root, dirs, files in os.walk(native_labels_path):
    total_images = pd.Series(files)
    for filename in files:
        full_path = os.path.join(root, filename)
        with open(full_path, 'r') as fp:
            lines = len(fp.readlines())
        
        PV_modules_in_each_image.append(lines)
        
solar_panels_in_each_image = pd.Series(PV_modules_in_each_image)
print("----------------------------------------------------------Dataset Statistics----------------------------------------------------------\n\n")
df = pd.DataFrame({'Image file' : total_images, 'number of solar panels' : solar_panels_in_each_image})
print(f"The total instances of solar panels in the dataset are: {solar_panels_in_each_image.sum()} \n\n")
print(df.head(), "\n\n")

label_counts = df['number of solar panels'].value_counts()

for num_panels, instances in label_counts.head().items():
    print(f"{num_panels} panels: {instances} instances")
    
print("...\n\n")

for num_panels, instances in label_counts.tail().items():
    print(f"{num_panels} panels: {instances} instances")
    

label_counts.plot(kind = 'hist', figsize = (8,4))
plt.xlabel("Number of Solar Panels per Image") 
plt.ylabel("Frequency")  
plt.title("Histogram of Solar Panel Counts per Image") 
plt.show()