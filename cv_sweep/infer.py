import glob
import os
import pandas as pd

from inference.yolo.inference_yolo_models import yolo_predict_tta
from preprocessing.dataset_yolo_format import save_dataset_in_yolo
from preprocessing.filter_boxes import get_filtered_train_df
from training.yolo.train_yolo_models import get_trained_yolo_models
from ultralytics import YOLO
# --- PREPROCESSING ---
# train_df = pd.read_csv('data/csv_files/Train.csv')

# Remove double boxes from the labels
# filtered_train_df = get_filtered_train_df(train_df, iou_threshold=0.3)

# Generate the YOLO dataset, ignoring the NEG labels
# save_dataset_in_yolo('data/img', filtered_train_df, 'data/yolo_ds')

# --- TRAINING ---
yolo_models = [YOLO("models/worthy_sweep3/train77/weights/best.pt"), YOLO("models/worthy_sweep3/train79/weights/best.pt"), YOLO("models/worthy_sweep3/train81/weights/best.pt"), YOLO("models/worthy_sweep3/train84/weights/best.pt"), YOLO("models/worthy_sweep3/train87/weights/best.pt")]

# FIXME: make these 2 return a model, yolo just loads one and detr is not exist
detr_csvs = ["csv_cv/detr_911/fold_1.csv", "csv_cv/detr_911/fold_2.csv", "csv_cv/detr_911/fold_3.csv", "csv_cv/detr_911/fold_4.csv", "csv_cv/detr_911/fold_5.csv"]

# --- validation 5 fold INFERENCE ---
#load the cv split
cv_split = pd.read_csv('data/csv_files/split_assignment.csv')
train_df = pd.read_csv('data/csv_files/Train.csv')

#loop through the 5 folds
for i in range(1, 6):
    print(f"Running fold {i}")
    print(len(yolo_models))
    yolo_model_fold = [yolo_models[i-1]]
    val_img_ids = cv_split[cv_split['Split'] == i]['Image_ID'].values
    val_df = train_df[train_df['Image_ID'].isin(val_img_ids)]

    val_images_paths = glob.glob('data/img/*.jpg')
    val_images_paths = [img for img in val_images_paths if os.path.basename(img) in val_df['Image_ID'].values]


    final_preds: dict[str, list[pd.DataFrame]] = {}

    for model in yolo_model_fold:
        yolo_preds = yolo_predict_tta(model, img_paths=val_images_paths)
        final_preds[f'yolo_{model.model_name}'] = yolo_preds

    for file in detr_csvs:
        # detr_preds = detr_tta_inference(model, image_paths=test_images_paths, tta_transforms=['flip'])
        detr_preds = [pd.read_csv(file)]
        final_preds['detr'] = detr_preds

    # Save the final predictions to disk
    os.makedirs('data/predictions', exist_ok=True)
    for model_name, preds in final_preds.items():
        for i, df in enumerate(preds):
            os.makedirs(f'data/predictions/SPLIT{i}{model_name[:3]}', exist_ok=True)
            df.to_csv(f'data/predictions/SPLIT{i}{model_name[:3]}/predictions_{i+1}.csv', index=False)















