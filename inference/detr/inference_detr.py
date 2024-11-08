from typing import List

import cv2
import pandas as pd
from tqdm import tqdm

from inference.tta import get_img_tta_augs
from inference.yolo.inference_yolo_models import yolo_predict


def detr_predict(model, image: cv2.Mat, augment: bool = False, conf: float = 0.0) -> List[dict]:
    pass


def detr_inference(model, images: List[cv2.Mat], conf: float = 0.0) -> pd.DataFrame:
    """
    Run YOLO model inference on a list of images.

    Args:
        model: YOLO model instance
        images: List of OpenCV images (cv2.Mat)
        conf: Confidence threshold

    Returns:
        DataFrame with predictions
    """
    all_preds = []
    for image in tqdm(images):
        preds = yolo_predict(model, image, conf=conf)
        for pred in preds:
            all_preds.append(pred)

    return pd.DataFrame(all_preds)


def detr_tta_inference(model, image_paths: List[str], tta_transforms: List[str] = None, conf: float = 0.0) -> List[
    pd.DataFrame]:
    """
    Run YOLO model inference with optional Test Time Augmentation (TTA).

    Args:
        model: YOLO model instance
        image_paths: List of paths to images
        tta_transforms: List of transforms to apply. Options: ['horizontal_flip', 'vertical_flip', 'both_flips']
        conf: Confidence threshold

    Returns:
        If tta_transforms is None or empty: DataFrame with predictions
        If tta_transforms is not empty: Dictionary of DataFrames with predictions for each augmentation
    """
    # Load images from file paths
    images = [cv2.imread(path) for path in image_paths]

    tta_images = get_img_tta_augs(images, tta_transforms)

    results = []
    for aug, aug_images in tta_images.items():
        results.append(detr_inference(model, aug_images, conf))

    return results
