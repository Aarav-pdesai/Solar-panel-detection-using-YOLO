# Solar Panel Detection using YOLO

## Overview
This project implements a solar panel detection system using YOLO (You Only Look Once) for object detection. The model is trained to identify solar panels in images and evaluate its performance using various accuracy metrics.

## Features
- Uses YOLO for solar panel detection
- Validates model performance on a test dataset
- Reports key accuracy metrics including mAP50, mAP50-95, precision, and recall


## Model Performance
The model is evaluated using:
- **mAP50**: 0.9834
- **mAP50-95**: 0.8878
- **Precision**: 0.9861
- **Recall**: 0.9584

## Directory Structure
```
Solar-panel-detection-using-YOLO/
│── assets/                     # Contains images or other asset files
│── dataset/                    # Dataset-related files
│── Implementing_fundamental_functions/  # TASK NUMBER 2
│   │── IoU_using_shapely.py
│   │── mAP50.py
│── model_training/             # Model training and evaluation scripts-------> TASK NUMBER 3
│   │── dataset_split.py         # Splitting dataset into train/test sets
│   │── fix_labels.py            # Fixes label formatting issues
│   │── model_train.py           # Script to train the YOLO model
│   │── model.py                 # Model evaluation and testing script
│── understanding_the_data/      # Data analysis and statistics---------------> TASK NUMBER 1
│   │── area_of_panels.py        # Computes panel areas from dataset
│   │── dataset_statistics.py    # Calculates dataset statistics
│── best.pt                      # Trained YOLO model weights
│── .gitignore                   # Git ignore file
│── my_personal_notes.txt        
│── readme.md                    # Project documentation
```

## Requirements
- Python 3.8+
- Ultralytics YOLO package
- OpenCV
- Torch


## Acknowledgments
This project is a selection task for an internship. 

## License
This project is for educational and research purposes only.

