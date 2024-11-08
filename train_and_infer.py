import glob
import os
import pandas as pd

from inference.yolo.inference_yolo_models import yolo_predict_tta
from preprocessing.dataset_yolo_format import save_dataset_in_yolo
from preprocessing.filter_boxes import get_filtered_train_df
from training.yolo.train_yolo_models import get_trained_yolo_models

# --- PREPROCESSING ---
# train_df = pd.read_csv('data/csv_files/Train.csv')

# Remove double boxes from the labels
# filtered_train_df = get_filtered_train_df(train_df, iou_threshold=0.3)

# Generate the YOLO dataset, ignoring the NEG labels
# save_dataset_in_yolo('data/img', filtered_train_df, 'data/yolo_ds')
# TODO ^^

# --- TRAINING ---
yolo_models = get_trained_yolo_models(
    glob.glob('config_files/yolo_train_config_files/*.yaml'),
    'data/yolo_ds/dataset.yaml'
)
# FIXME: make these 2 return a model, yolo just loads one and detr is not exist
detr_models = [None]

# --- INFERENCE ---
test_df = pd.read_csv('data/csv_files/Test.csv')
test_images_paths = glob.glob('data/img/*.jpg')
test_image_ids = [os.path.basename(img) for img in test_images_paths]
test_images_paths = [img for img in test_images_paths if os.path.basename(img) in test_df['Image_ID'].values]

final_preds: dict[str, list[pd.DataFrame]] = {}

for model in yolo_models:
    yolo_preds = yolo_predict_tta(model, img_paths=test_images_paths)
    final_preds[f'yolo_{model.model_name}'] = yolo_preds

for model in detr_models:
    # detr_preds = detr_tta_inference(model, image_paths=test_images_paths, tta_transforms=['flip'])
    detr_preds = [pd.read_csv('data/processed_predictions_new.csv')]
    final_preds['detr'] = detr_preds

# Save the final predictions to disk
os.makedirs('data/predictions', exist_ok=True)
for model_name, preds in final_preds.items():
    for i, df in enumerate(preds):
        os.makedirs(f'data/predictions/{model_name[:3]}', exist_ok=True)
        df.to_csv(f'data/predictions/{model_name[:3]}/predictions_{i+1}.csv', index=False)