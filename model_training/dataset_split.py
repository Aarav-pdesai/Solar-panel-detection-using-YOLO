# first we must split the dataset according to the structre required by YOLO models

import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import glob

images_path = "Solar-panel-detection-using-YOLO/assets/images"
labels_path = "Solar-panel-detection-using-YOLO/assets/labels"

for tif_file in glob.glob(os.path.join(images_path, "*.tif")):
    with Image.open(tif_file) as img:
        jpg_file = tif_file.replace(".tif", ".jpg")
        img.convert("RGB").save(jpg_file, "JPEG")


output_dir = "Solar-panel-detection-using-YOLO/dataset"
os.makedirs(os.path.join(output_dir, "images/train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images/val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images/test"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/test"), exist_ok=True)

image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))])
label_files = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in image_files]

# (80-10-10 split)
train_images, temp_images, train_labels, temp_labels = train_test_split( # temp because the 20% split is not going to be used directly anywhere later
    image_files, label_files, test_size=0.2, random_state=1
)
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=1
)

def move_files(file_list, src_dir, dest_dir):
    for file_name in file_list:
        src_path = os.path.join(src_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)

move_files(train_images, images_path, os.path.join(output_dir, "images/train"))
move_files(val_images, images_path, os.path.join(output_dir, "images/val"))
move_files(test_images, images_path, os.path.join(output_dir, "images/test"))

move_files(train_labels, labels_path, os.path.join(output_dir, "labels/train"))
move_files(val_labels, labels_path, os.path.join(output_dir, "labels/val"))
move_files(test_labels, labels_path, os.path.join(output_dir, "labels/test"))
