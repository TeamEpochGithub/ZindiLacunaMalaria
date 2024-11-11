from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = ['WBC', 'Trophozoite', 'NEG']

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    y1, x1, y2, x2 = box1
    y1_, x1_, y2_, x2_ = box2

    xi1, yi1 = max(x1, x1_), max(y1, y1_)
    xi2, yi2 = min(x2, x2_), min(y2, y2_)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def calculate_ap50_per_class(pred_boxes, gt_boxes):
    tp = []
    fp = []
    scores = []
    num_gt = sum(len(boxes) for boxes in gt_boxes.values())
    print(num_gt)

    for image_id, pred_image_boxes in pred_boxes.items():
        gt_image_boxes = gt_boxes.get(image_id, [])

        if not gt_image_boxes:
            fp.extend([1] * len(pred_image_boxes))
            tp.extend([0] * len(pred_image_boxes))

            scores.extend([box[2] for box in pred_image_boxes])
            continue

        pred_image_boxes = sorted(pred_image_boxes, key=lambda x: x[2], reverse=True)
        gt_matched = [False] * len(gt_image_boxes)

        # to_consider = pred_image_boxes[:len(gt_image_boxes)]
        for pred_box in pred_image_boxes:
            max_iou = 0
            max_gt_idx = -1

            for j, gt_box in enumerate(gt_image_boxes):
                if gt_matched[j]:
                    continue

                iou = calculate_iou(pred_box[3:], gt_box[2:])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = j

            if max_iou >= 0.5:
                tp.append(1)
                fp.append(0)
                gt_matched[max_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)
            scores.append(pred_box[2])

    # Sort by score
    indices = np.argsort(-np.array(scores))
    tp = np.array(tp)[indices]
    fp = np.array(fp)[indices]

    # Compute precision recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / num_gt

    # Compute AP
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([1], precision, [0]))

    # Ensure precision curve is non-increasing
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return ap


def ap_per_class(pred_df, gt_df):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        pred_df (pd.DataFrame): DataFrame containing predicted boxes.
        gt_df (pd.DataFrame): DataFrame containing ground truth boxes.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (dict, optional): Dict of class names to plot PR curves. Defaults to an empty dict.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        (tuple): A tuple of arrays and unique classes, where:
            ap (np.ndarray): Average precision for each class. Shape: (nc,).
            unique_classes (np.ndarray): An array of unique classes that have data. Shape: (nc,).
    """
    pred_boxes = defaultdict(lambda: defaultdict(list))
    gt_boxes = defaultdict(lambda: defaultdict(list))

    for _, row in pred_df.iterrows():
        pred_boxes[row['class']][row['Image_ID']].append(
            (row['Image_ID'], row['class'], row['confidence'], row['ymin'], row['xmin'], row['ymax'], row['xmax']))

    for _, row in gt_df.iterrows():
        gt_boxes[row['class']][row['Image_ID']].append(
            (row['Image_ID'], row['class'], row['ymin'], row['xmin'], row['ymax'], row['xmax']))

    nc = len(CLASSES)

    ap = np.zeros(nc)
    for ci, c in enumerate(CLASSES):
        ap[ci] = calculate_ap50_per_class(pred_boxes[c], gt_boxes[c])

    return ap


def calculate_map50(pred_df, gt_df, verbose=True):
    """Calculate MAP50 score"""
    image_ids = pred_df['Image_ID'].unique().tolist()
    filtered_gt_df = gt_df[gt_df['Image_ID'].isin(image_ids)]
    ap = ap_per_class(pred_df, filtered_gt_df)

    # print the AP for each class
    if verbose:
        for i, c in enumerate(CLASSES):
            print(f"AP for class {c}: {ap[i]}")

    return ap


def main():
    # Mock data for demonstration
    df = pd.read_csv("data/csv_files/Train.csv")

    # For demonstration, we're using the same DataFrame for both ground truth and predictions
    map_50 = calculate_map50(df, df)
    print(map_50)

    print(f"mAP@50: {np.mean(map_50):.4f}")

if __name__ == "__main__":
    main()
