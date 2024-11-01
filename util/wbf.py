import glob
import os
from typing import Tuple, List
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 4)

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 4)


def calculate_iou(box1: Tuple[float, float, float, float],
                  box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate IoU between two boxes
    Args:
        box1: (ymin, xmin, ymax, xmax)
        box2: (ymin, xmin, ymax, xmax)
    Returns:
        IoU score
    """
    # Calculate intersection coordinates
    ymin_inter = max(box1[0], box2[0])
    xmin_inter = max(box1[1], box2[1])
    ymax_inter = min(box1[2], box2[2])
    xmax_inter = min(box1[3], box2[3])

    # Calculate area of intersection
    inter_area = max(0, xmax_inter - xmin_inter) * max(0, ymax_inter - ymin_inter)

    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate Union area
    union_area = box1_area + box2_area - inter_area

    # Return IoU
    return inter_area / union_area if union_area > 0 else 0

def process_single_image(image_data: tuple) -> pd.DataFrame:
    """
    Process a single image's predictions using WBF
    Args:
        image_data: Tuple containing (image_predictions, num_models, iou_thresh)
    Returns:
        DataFrame with processed predictions for this image
    """
    image_predictions, num_models, iou_thresh = image_data
    final_df = pd.DataFrame(columns=['Image_ID', 'class', 'confidence',
                                     'ymin', 'xmin', 'ymax', 'xmax'])

    while len(image_predictions) > 0:
        # Get current prediction
        current_pred = image_predictions.iloc[0]
        current_box = (current_pred['ymin'], current_pred['xmin'],
                       current_pred['ymax'], current_pred['xmax'])

        # Initialize lists for matching boxes
        matching_boxes = []
        matching_indices = []
        matching_confidences = []

        # Compare with other predictions
        for idx, row in image_predictions.iterrows():
            if idx == 0:
                continue

            compare_box = (row['ymin'], row['xmin'], row['ymax'], row['xmax'])
            iou = calculate_iou(current_box, compare_box)

            if (iou > iou_thresh and current_pred['class'] == row['class']):
                matching_boxes.append(compare_box)
                matching_indices.append(idx)
                matching_confidences.append(row['confidence'])

        matching_boxes.append(current_box)
        matching_confidences.append(current_pred['confidence'])

        # Calculate weighted average
        weights = np.array(matching_confidences)
        weights = weights / weights.sum()

        boxes_array = np.array(matching_boxes)
        weighted_box = np.average(boxes_array, weights=weights, axis=0)

        # Create new row with weighted box
        new_row = current_pred.copy()
        new_row[['ymin', 'xmin', 'ymax', 'xmax']] = weighted_box
        new_row['confidence'] = np.max(matching_confidences) * (len(matching_boxes) / num_models)

        # Remove processed predictions
        image_predictions = image_predictions.drop(
            matching_indices + [0]).reset_index(drop=True)

        final_df = pd.concat([final_df, pd.DataFrame([new_row])],
                             ignore_index=True)

    return final_df

def weighted_boxes_fusion(pass_df: bool,
                          input_path: str,
                          output_path: str,
                          conf_thresh: float = 0.5,
                          iou_thresh: float = 0.5,
                          n_processes: int = None) -> pd.DataFrame:
    """
    Implement parallelized Weighted Boxes Fusion algorithm
    Args:
        pass_df: Whether to return the DataFrame instead of saving to file
        input_path: Path to folder containing prediction CSV files
        output_path: Path to save final predictions
        conf_thresh: Confidence threshold for filtering predictions
        iou_thresh: IoU threshold for box fusion
        n_processes: Number of processes to use (defaults to CPU count)
    Returns:
        DataFrame with final predictions if pass_df is True
    """
    if n_processes is None:
        n_processes = mp.cpu_count()

    # Get all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_path, "*.csv"))
    num_models = len(csv_files)

    # Read and combine all CSV files
    all_predictions = pd.concat([pd.read_csv(f) for f in csv_files],
                                ignore_index=True)

    # Handle negative cases
    neg_rows = all_predictions[all_predictions['class'] == 'neg']
    final_df = pd.DataFrame(columns=['Image_ID', 'class', 'confidence',
                                     'ymin', 'xmin', 'ymax', 'xmax'])
    final_df = pd.concat([final_df, neg_rows], ignore_index=True)
    all_predictions = all_predictions[~(all_predictions['class'] == 'neg')]

    # Filter low confidence predictions
    low_conf_mask = all_predictions['confidence'] < conf_thresh
    final_df = pd.concat([final_df, all_predictions[low_conf_mask]],
                         ignore_index=True)
    all_predictions = all_predictions[~low_conf_mask].reset_index(drop=True)

    # Prepare data for parallel processing
    image_groups = []
    for image_id in all_predictions['Image_ID'].unique():
        image_predictions = all_predictions[
            all_predictions['Image_ID'] == image_id].reset_index(drop=True)
        image_groups.append((image_predictions, num_models, iou_thresh))

    # Process images in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, image_groups),
            total=len(image_groups),
            desc="Processing images"
        ))

    # Combine results
    final_df = pd.concat([final_df] + results, ignore_index=True)

    # Sort by Image_ID and confidence
    final_df = final_df.sort_values(['Image_ID', 'confidence'],
                                    ascending=[True, False]).reset_index(drop=True)

    if pass_df:
        return final_df

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save final predictions
    final_df.to_csv(output_path, index=False)

def weighted_boxes_fusion_df(
                          df: pd.DataFrame,
                          conf_thresh: float = 0.5,
                          iou_thresh: float = 0.5,
                          n_processes: int = None) -> pd.DataFrame:
    """
    Implement parallelized Weighted Boxes Fusion algorithm
    Args:
        df: DataFrame containing predictions
        conf_thresh: Confidence threshold for filtering predictions
        iou_thresh: IoU threshold for box fusion
        n_processes: Number of processes to use (defaults to CPU count)
    Returns:
        DataFrame with final predictions 
    """
    if n_processes is None:
        n_processes = mp.cpu_count()

    # Read and combine all CSV files
    all_predictions = df
    # Handle negative cases
    
    final_df = pd.DataFrame(columns=['Image_ID', 'class', 'confidence',
                                     'ymin', 'xmin', 'ymax', 'xmax'])
    
    # Filter low confidence predictions
    low_conf_mask = all_predictions['confidence'] < conf_thresh
    final_df = pd.concat([final_df, all_predictions[low_conf_mask]],
                         ignore_index=True)
    all_predictions = all_predictions[~low_conf_mask].reset_index(drop=True)

    # Prepare data for parallel processing
    image_groups = []
    for image_id in all_predictions['Image_ID'].unique():
        image_predictions = all_predictions[
            all_predictions['Image_ID'] == image_id].reset_index(drop=True)
        image_groups.append((image_predictions, 1, iou_thresh))

    # Process images in parallel
    with mp.Pool(processes=n_processes) as pool:
     results = list(pool.imap(process_single_image, image_groups))

    # Combine results
    final_df = pd.concat([final_df] + results, ignore_index=True)

    # Sort by Image_ID and confidence
    final_df = final_df.sort_values(['Image_ID', 'confidence'],
                                    ascending=[True, False]).reset_index(drop=True)

    return final_df

def apply_wbf_to_df(df: pd.DataFrame, conf_thresh: float = 0.5, iou_thresh: float = 0.5):
    return weighted_boxes_fusion_df(df, conf_thresh=conf_thresh, iou_thresh=iou_thresh)


if __name__ == "__main__":
    input_path = "../src/wbf_submission_files/"
    output_path = "../data/csv_files/wbf_submission_predictions.csv"
    weighted_boxes_fusion(
        pass_df=False,
        input_path=input_path,
        output_path=output_path,
        conf_thresh=0.1541,
        iou_thresh=0.6715,
        n_processes=mp.cpu_count()  # Use all available CPU cores
    )