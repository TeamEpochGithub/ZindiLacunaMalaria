import glob

import pandas as pd

from preprocessing.dataset_yolo_format import save_dataset_in_yolo
from preprocessing.filter_boxes import get_filtered_train_df
from training.yolo.train_yolo_models import get_trained_yolo_models

train_df = pd.read_csv('data/csv_files/Train.csv')

# remove double boxes from the labels
filtered_train_df = get_filtered_train_df(train_df, iou_threshold=0.3)

# generate the YOLO dataset, ignoring the NEG labels
save_dataset_in_yolo('data/img', filtered_train_df, 'data/yolo_ds')

yolo_models = get_trained_yolo_models(glob.glob('parameters/yolo_config_files/*.yaml'), 'data/yolo_ds/dataset.yaml')