import numpy as np
import random
def ap_voc_11(precision, recall):
    recall_levels = np.linspace(0, 1, 11) # only difference is in number of levels, 11 in VOC, 101 in COCO
    ap = 0
    for recall_level in recall_levels:
        max_precision = np.max(precision[recall >= recall_level]) if np.any(recall >= recall_level) else 0
        ap += max_precision
    ap /= len(recall_levels)
    return ap

def ap_coco_101(precision, recall):
    recall_levels = np.linspace(0, 1, 101)#more granular
    ap = 0
    for recall_level in recall_levels:
        max_precision = np.max(precision[recall >= recall_level]) if np.any(recall >= recall_level) else 0
        ap += max_precision
    ap /= len(recall_levels)
    return ap

def ap_area_under_curve(precision, recall):
    sorted_indices = np.argsort(recall)
    recall = recall[sorted_indices]
    precision = precision[sorted_indices]
    ap = np.trapz(precision, recall)  
    return ap

def compute_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def generate_random_boxes(num_boxes, image_size, box_size):
    boxes = []
    for _ in range(num_boxes):
        x_min = random.randint(0, image_size[0] - box_size[0])
        y_min = random.randint(0, image_size[1] - box_size[1])
        x_max = x_min + box_size[0]
        y_max = y_min + box_size[1]
        boxes.append([x_min, y_min, x_max, y_max])
    return np.array(boxes)

ground_truth_boxes = [generate_random_boxes(10, (100, 100), (20, 20)) for _ in range(10)]
predicted_boxes = [generate_random_boxes(10, (100, 100), (20, 20)) for _ in range(10)]

def compute_precision_recall(ground_truth_boxes, predicted_boxes, iou_threshold=0.5):
    tp = np.zeros(len(predicted_boxes))
    fp = np.zeros(len(predicted_boxes))
    detected = set()

    for i, pred_box in enumerate(predicted_boxes):
        max_iou = 0
        matched_gt_idx = -1

        for j, gt_box in enumerate(ground_truth_boxes):
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
    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / len(ground_truth_boxes)

    return precision, recall

for i in range(len(ground_truth_boxes)):
    precision, recall = compute_precision_recall(ground_truth_boxes[i], predicted_boxes[i], iou_threshold=0.5)
    ap_voc11 = ap_voc_11(precision, recall)
    ap_coco101 = ap_coco_101(precision, recall)
    ap_auc_pr = ap_area_under_curve(precision, recall)

    print(f"Image {i + 1}:")
    print(f"  AP50 (VOC 11-point interpolation): {ap_voc11}")
    print(f"AP50 (COCO 101-point interpolation): {ap_coco101}")
    print(f"                      AP50 (AUC-PR): {ap_auc_pr}")


