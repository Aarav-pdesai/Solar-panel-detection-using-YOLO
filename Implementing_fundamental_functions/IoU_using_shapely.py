from shapely.geometry import box
import numpy as np
from supervision.detection.utils import box_iou_batch

def yolo_to_corners(center_x, center_y, width, height):
    xmin = center_x - (width / 2)
    ymin = center_y - (height / 2)
    xmax = center_x + (width / 2)
    ymax = center_y + (height / 2)
    return xmin, ymin, xmax, ymax

def compute_iou_with_shapely(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = yolo_to_corners(*box1)
    xmin2, ymin2, xmax2, ymax2 = yolo_to_corners(*box2)
    
    rect1 = box(xmin1, ymin1, xmax1, ymax1)
    rect2 = box(xmin2, ymin2, xmax2, ymax2)

    intersection_area = rect1.intersection(rect2).area
    union_area = rect1.union(rect2).area

    iou = intersection_area / union_area 
    return iou

GT_box1 = [3, 3, 6, 6]  # example for verification
box2 = [2.5, 2.5,5, 5]  

iou_shapely = compute_iou_with_shapely(GT_box1, box2)# IoU  with Shapely
print(f"IoU using Shapely: {iou_shapely:.4f}")

box1_corners = np.array([yolo_to_corners(*GT_box1)]) #Convert to corner format
box2_corners = np.array([yolo_to_corners(*box2)])

ious = box_iou_batch(box1_corners, box2_corners)
iou_supervision = ious[0, 0]
print(f"IoU using supervision: {iou_supervision:.4f}")
