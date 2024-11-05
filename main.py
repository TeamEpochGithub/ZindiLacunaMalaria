import glob
import os
from typing import List
import pandas as pd

from inference.detr.inference_detr import detr_tta_inference
from inference.yolo.inference_yolo_models import yolo_tta_inference
from postprocessing.tta_postprocessor import postprocess
from preprocessing.dataset_yolo_format import save_dataset_in_yolo
from preprocessing.filter_boxes import get_filtered_train_df
from training.detr.train_detr_models import get_trained_detr_models
from training.yolo.train_yolo_models import get_trained_yolo_models

# Load training data
train_df = pd.read_csv('data/csv_files/Train.csv')

# Remove double boxes from the labels
filtered_train_df = get_filtered_train_df(train_df, iou_threshold=0.3)

# Generate the YOLO dataset, ignoring the NEG labels
save_dataset_in_yolo('data/img', filtered_train_df, 'data/yolo_ds')

# Train the models
yolo_models = get_trained_yolo_models(
    glob.glob('config_files/yolo_train_config_files/*.yaml'),
    'data/yolo_ds/dataset.yaml'
)
detr_models = get_trained_detr_models(...)

# Get test image paths
test_df = pd.read_csv('data/csv_files/Test.csv')
test_images_paths = glob.glob('data/img/*.jpg')
test_image_ids = [os.path.basename(img) for img in test_images_paths]
test_images_paths = [img for img in test_images_paths if os.path.basename(img) in test_df['Image_ID'].values]

# Get predictions for YOLO using TTA
predictions: List[pd.DataFrame] = []
for model in yolo_models:
    yolo_preds = yolo_tta_inference(model, image_paths=test_images_paths, tta_transforms=['flip', 'rotate'])
    for i, pred in enumerate(yolo_preds):
        yolo_preds[i] = postprocess(pred, ...)
    yolo_preds_processed = postprocess(yolo_preds, ...)
    predictions += yolo_preds_processed

# Get predictions for DETR using TTA
for model in detr_models:
    detr_preds = detr_tta_inference(model, image_paths=test_images_paths, tta_transforms=['flip', 'rotate'])
    for i, pred in enumerate(detr_preds):
        detr_preds[i] = postprocess(pred, ...)
    detr_preds_processed = postprocess(detr_preds, ...)
    predictions += detr_preds_processed

# Postprocess the final predictions
final_pred_df = postprocess(predictions, ...)