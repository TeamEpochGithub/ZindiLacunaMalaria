import glob
import os

import pandas as pd

from inference.yolo.inference_yolo_models import inference_yolo_model
from preprocessing.dataset_yolo_format import save_dataset_in_yolo
from preprocessing.filter_boxes import get_filtered_train_df
from training.yolo.train_yolo_models import get_trained_yolo_models

train_df = pd.read_csv('data/csv_files/Train.csv')

# remove double boxes from the labels
filtered_train_df = get_filtered_train_df(train_df, iou_threshold=0.3)

# generate the YOLO dataset, ignoring the NEG labels
save_dataset_in_yolo('data/img', filtered_train_df, 'data/yolo_ds')

# train the NEG classifier
# neg_model =

# train the yolo models
yolo_models = get_trained_yolo_models(glob.glob('parameters/yolo_config_files/*.yaml'), 'data/yolo_ds/dataset.yaml')

# get test image paths
test_df = pd.read_csv('data/csv_files/Test.csv')
test_images = glob.glob('data/img/*.jpg')
test_image_ids = [os.path.basename(img) for img in test_images]
test_images = [img for img in test_images if os.path.basename(img) in test_df['Image_ID'].values]

predictions = []

# get predictions for yolo using TTA (Test Time Augmentation)
for model in yolo_models:
    predictions.append(inference_yolo_model(model, image_paths=test_images, tta=True))


# ensemble all predictions and postprocess