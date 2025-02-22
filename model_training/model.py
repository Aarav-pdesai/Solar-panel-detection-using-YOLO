import os
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from supervision.metrics import MeanAveragePrecision
from supervision.detection.core import Detections


import numpy as np
import pandas as pd

model = YOLO("runs/detect/train/weights/best.pt")  
# results = model.train(data="C:/Users/aarav/OneDrive/Desktop/IIT_GN_TASK/Solar-panel-detection-using-YOLO/dataset.yaml", epochs=15, imgsz=640, val=True)

# TASK 1: Show that validation loss is converged
df = pd.read_csv("C:/Users/aarav/OneDrive/Desktop/IIT_GN_TASK/runs/detect/train/results.csv")  

# Compute total loss (sum of relevant columns)
df["train/loss"] = df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]
df["val/loss"] = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]

# Plot training and validation loss
plt.plot(df["epoch"], df["train/loss"], label="Training Loss")
plt.plot(df["epoch"], df["val/loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss Curve")
plt.show()


# TASK 2:  visualize the ground truth and predicted bounding boxes on 3-4 random samples
test_images_dir = "Solar-panel-detection-using-YOLO/dataset/images/test"
test_labels_dir = "Solar-panel-detection-using-YOLO/dataset/labels/test"
test_images = os.listdir(test_images_dir)
random_samples = random.sample(test_images, 3)


def load_ground_truth_boxes_for_rand_samples(label_path, img_w, img_h):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                _, x_center, y_center, width, height = map(float, line.split())
                x_min = int((x_center - width / 2) * img_w)
                y_min = int((y_center - height / 2) * img_h)
                x_max = int((x_center + width / 2) * img_w)
                y_max = int((y_center + height / 2) * img_h)
                boxes.append((x_min, y_min, x_max, y_max))
    return boxes

# function to draw ground truth and prediction boxes
for img_file in random_samples:
    img_path = os.path.join(test_images_dir, img_file)
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    label_file = os.path.join(test_labels_dir, img_file.replace(".jpg", ".txt"))
    ground_truth_boxes = load_ground_truth_boxes_for_rand_samples(label_file, img_w, img_h)

    results = model.predict(img_path, show=False, verbose = False)
    predicted_boxes = results[0].boxes.xyxy

    #ground truth boxes in green
    for box in ground_truth_boxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, "Ground Truth", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # predicted boxes in red
    for box in predicted_boxes:
        x_min, y_min, x_max, y_max = map(int, box[:4].tolist())
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(img, "Prediction", (x_max, y_max + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Ground Truth vs Predictions: {img_file}")
    plt.axis("off")
    plt.show()


# TASK 3: mAP50, precision, recall and F1 score
def compute_iou(box1, box2):
    if not (isinstance(box1, (list, tuple, np.ndarray)) and len(box1) == 4):
        raise ValueError(f"Invalid box1 format: {box1}")
    if not (isinstance(box2, (list, tuple, np.ndarray)) and len(box2) == 4):
        raise ValueError(f"Invalid box2 format: {box2}")
    box1 = np.array(box1)
    box2 = np.array(box2)

    # Compute intersection coordinates
    x1_inter = np.maximum(box1[0], box2[0])
    y1_inter = np.maximum(box1[1], box2[1])
    x2_inter = np.minimum(box1[2], box2[2])
    y2_inter = np.minimum(box1[3], box2[3])

    # Compute intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Compute areas of the bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute union area
    union_area = box1_area + box2_area - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    # Compute IoU
    return inter_area / union_area

# Function to compute precision and recall
def compute_precision_recall(ground_truth_boxes, predicted_boxes, iou_threshold=0.5):
    tp = np.zeros(len(predicted_boxes))
    fp = np.zeros(len(predicted_boxes))
    detected = set()

    for i, pred_box in enumerate(predicted_boxes):
        if isinstance(pred_box, tuple):
            pred_box = pred_box[0]  # Extract bounding box coordinates

        max_iou = 0
        matched_gt_idx = -1

        for j, gt_box in enumerate(ground_truth_boxes):
            if isinstance(gt_box, tuple):
                gt_box = gt_box[0]  # Extract bounding box coordinates
            
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_threshold and iou > max_iou and j not in detected:
                max_iou = iou
                matched_gt_idx = j

        if matched_gt_idx != -1:
            tp[i] = 1  # True positive
            detected.add(matched_gt_idx)
        else:
            fp[i] = 1  # False positive
            
    tp_sum = np.cumsum(tp)
    fp_sum = np.cumsum(fp)
    precision = tp_sum / (tp_sum + fp_sum + 1e-6)  # Avoid division by zero
    recall = tp_sum / (len(ground_truth_boxes) + 1e-6)  # Avoid division by zero

    return precision, recall


def ap_area_under_curve(precision, recall):
    sorted_indices = np.argsort(recall)
    recall = recall[sorted_indices]
    precision = precision[sorted_indices]
    ap = np.trapz(precision, recall)  # Numerical integration
    return ap

def load_ground_truth_boxes(labels_dir):
    ground_truth_boxes = []

    for label_file in os.listdir(labels_dir):
        boxes = []
        class_ids = []

        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.split())
                x_min = x_center - (width / 2)
                y_min = y_center - (height / 2)
                x_max = x_center + (width / 2)
                y_max = y_center + (height / 2)

                boxes.append([x_min, y_min, x_max, y_max])
                class_ids.append(int(class_id))

        # Convert to Detections object
        ground_truth_boxes.append(Detections(
            xyxy=np.array(boxes), 
            class_id=np.array(class_ids)
        ))

    return ground_truth_boxes

# Load ground truth annotations
all_ground_truth_boxes = load_ground_truth_boxes("Solar-panel-detection-using-YOLO/dataset/labels/test")

def perform_predictions(images_dir, model):
    predicted_boxes = []

    for img_file in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)

        # Perform inference using the trained model
        results = model.predict(img, conf=0.1, iou=0.6)  

        if not results or not results[0].boxes:  # Ensure results exist
            predicted_boxes.append(Detections.empty())  # Empty prediction for this image
            continue
        
        img_height, img_width = results[0].orig_shape 
        
        boxes = results[0].boxes.xyxy.clone()  # Clone the tensor to allow modifications
        boxes[:, [0, 2]] /= img_width   # Normalize x-coordinates
        boxes[:, [1, 3]] /= img_height  # Normalize y-coordinates

        # Convert to numpy
        boxes = boxes.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs

        # Create a Detections object
        detections = Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids
        )
        predicted_boxes.append(detections)

    return predicted_boxes


# Perform predictions using the trained model
all_predicted_boxes = perform_predictions("Solar-panel-detection-using-YOLO/dataset/images/test", model)

print("Sample Ground Truth Boxes:", all_ground_truth_boxes[:2])
print("Sample Predicted Boxes:", all_predicted_boxes[:2])


#mAP using supervision.metrics
map50_supervision = MeanAveragePrecision()
for pred, gt in zip(all_predicted_boxes, all_ground_truth_boxes):
    map50_supervision.update(pred, gt)
map_result = map50_supervision.compute()
print(f"Supervision mAP50: {map_result.map50}")

# Calculate mAP50 using custom Area Under Curve (AUC) method
map_auc_results = []
for i in range(len(all_ground_truth_boxes)):
    precision, recall = compute_precision_recall(all_ground_truth_boxes[i], all_predicted_boxes[i], iou_threshold=0.5)
    map_auc_results.append(ap_area_under_curve(precision, recall))

custom_map50 = np.mean(map_auc_results)
print(f"Custom mAP50 using AUC: {custom_map50}")

# Function to compute Precision, Recall, and F1-score
def compute_metrics(ground_truth_boxes, predicted_boxes, iou_threshold, confidence_threshold):
    tp, fp, fn = 0, 0, 0

    for gt_boxes, pred_boxes in zip(ground_truth_boxes, predicted_boxes):
        # Filter predicted boxes by confidence threshold
        filtered_pred_boxes = [box for box in pred_boxes if box[4] is not None and box[4] >= confidence_threshold]

        # Track matched ground truth boxes
        matched_gt = set()

        # Calculate TP and FP
        for pred_box in filtered_pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for idx, gt_box in enumerate(gt_boxes):
                if idx not in matched_gt:
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        # Calculate FN (ground truth boxes not matched)
        fn += len(gt_boxes) - len(matched_gt)

    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score

# Create a table of Precision, Recall, and F1-scores with IoU thresholds and confidence thresholds
iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
confidence_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

# Initialize results table
results_table = []

# Compute metrics for each combination of IoU and confidence thresholds
for iou_thresh in iou_thresholds:
    row = []
    for conf_thresh in confidence_thresholds:
        precision, recall, f1_score = compute_metrics(all_ground_truth_boxes, all_predicted_boxes, iou_thresh, conf_thresh)
        row.append((precision, recall, f1_score))
    results_table.append(row)

# Display the results as a table
for i, row in enumerate(results_table):
    print(f"IoU Threshold: {iou_thresholds[i]}")
    for j, (precision, recall, f1_score) in enumerate(row):
        print(f"  Confidence {confidence_thresholds[j]} -> Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_score:.2f}")

