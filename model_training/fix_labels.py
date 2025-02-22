import os
import glob

label_paths = ["Solar-panel-detection-using-YOLO/dataset/labels/train", "Solar-panel-detection-using-YOLO/dataset/labels/val", "Solar-panel-detection-using-YOLO/dataset/labels/test"]  # Adjust path as needed

for label_dir in label_paths:
    for label_file in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(label_file, "r") as f:
            lines = f.readlines()

        fixed_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                parts[0] = "0"  # Change class index to 0
                fixed_lines.append(" ".join(parts))

        with open(label_file, "w") as f:
            f.write("\n".join(fixed_lines))

print("Dataset labels fixed: All classes are now set to 0.")
