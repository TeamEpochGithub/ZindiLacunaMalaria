import glob
import os
import pandas as pd
from time import time

from inference.inference_neg_model import inference
from inference.yolo.inference_yolo_models import yolo_predict_tta
from inference.detr.inference_detr_models import detr_predict_tta
from preprocessing.dataset_yolo_format import save_dataset_in_yolo
from preprocessing.filter_boxes import get_filtered_train_df
from training.yolo.train_yolo_models import get_trained_yolo_models
from training.train_neg_model import train_model, initialize_model
from training.detr.train_detr_models import get_trained_detr_models
from util.yaml_structuring import create_structured_config, load_yaml_config
from postprocess import run_postprocessing
from util.save import add_negs_to_submission
import torch

train_start = time()
# --- PREPROCESSING ---
train_df = pd.read_csv('data/csv_files/Train.csv')

# Remove double boxes from the labels
filtered_train_df = get_filtered_train_df(train_df, iou_threshold=0.3)

# Generate the YOLO dataset, ignoring the NEG labels
save_dataset_in_yolo('data/img', filtered_train_df, 'data/yolo_ds')
# TODO ^^

# --- TRAINING ---
yolo_models = get_trained_yolo_models(
    glob.glob('config_files/yolo_train_config_files/*.yaml'),
    'data/yolo_ds/dataset.yaml'
)
detr_models = get_trained_detr_models(
    glob.glob('config_files/detr_train_config_files/*.yaml'),
    'data/yolo_ds/dataset.yaml'
)
if "neg_model.pth" not in os.listdir("models"):
    neg_model = train_model('data/img', 'data/csv_files/Train.csv', 2)
    # Save the model's state_dict
    torch.save(neg_model.state_dict(), 'models/neg_model.pth')
else:
    # Initialize the model architecture
    neg_model = initialize_model()
    # Load the saved state_dict
    neg_model.load_state_dict(torch.load("models/neg_model.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neg_model.to(device)
train_time = time() - train_start
print(f"Training took {train_time / 3600:.2f} hours")

inference_start = time()
# --- INFERENCE ---
# save our NEG predictions in a csv
result = inference(neg_model, 'data/img', 'data/csv_files/Test.csv')
neg_preds = []
for id, pred in result.items():
    neg_preds.append([id, pred.replace("POS", "NON_NEG")])
pd.DataFrame(neg_preds, columns=['Image_ID', 'class']).to_csv('data/csv_files/NEG_OR_NOT2.csv', index=False)

test_df = pd.read_csv('data/csv_files/Test.csv')
test_images_paths = glob.glob('data/img/*.jpg')
test_image_ids = [os.path.basename(img) for img in test_images_paths]
test_images_paths = [img for img in test_images_paths if os.path.basename(img) in test_df['Image_ID'].values]

final_preds: dict[str, list[pd.DataFrame]] = {}

# Get predictions for YOLO using TTA
for model in yolo_models:
    yolo_preds = yolo_predict_tta(model, img_paths=test_images_paths)
    final_preds[f'yolo'] = yolo_preds

# Get predictions for DETR using TTA
detr_config_files = glob.glob('config_files/detr_train_config_files/*.yaml')
for model, config_file in zip(detr_models, detr_config_files):
    detr_preds = detr_predict_tta(model, config_file, img_paths=test_images_paths)
    final_preds['detr'] = detr_preds

# Save the final predictions to disk
os.makedirs('data/predictions', exist_ok=True)
for model_name, preds in final_preds.items():
    for i, df in enumerate(preds):
        df.to_csv(f'data/predictions/{model_name[:3]}_predictions_{i + 1}.csv', index=False)

# --- POSTPROCESSING ---

config_file = "parameters/postprocessing_config_files/vocal_sweep_53.yaml"
config = load_yaml_config(config_file)
param_config = create_structured_config(config["parameters"])

print(final_preds.keys())
detr_tta_files = final_preds['detr']
yolo_tta_files = final_preds['yolo']

all_df = run_postprocessing(param_config, 1, yolo_tta_files, detr_tta_files)

# Add NEG predictions to the final submission
os.makedirs('submissions', exist_ok=True)
neg_all_df = pd.read_csv("data/csv_files/NEG_OR_NOT2.csv")
final_submission_df = add_negs_to_submission(df=all_df, neg_csv="data/csv_files/NEG_OR_NOT2.csv", test_csv="data/csv_files/Test.csv")
final_submission_df.to_csv("submissions/final_submission.csv", index=False)

inference_time = time() - inference_start
print(f"Inference took {inference_time / 3600:.2f} hours")