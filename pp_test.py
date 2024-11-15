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
from util.ensemble import DualEnsemble
import torch

def ensemble_wbf_only(tta_files):
    detr_tta_files = ["data/predictions/"+f for f in tta_files if "det" in f and "neg" not in f]
    yolo_tta_files = ["data/predictions/"+f for f in tta_files if "yol" in f]

    detr_tta_dfs = [pd.read_csv(f) for f in detr_tta_files]
    yolo_tta_dfs = [pd.read_csv(f) for f in yolo_tta_files]


    yolo_ens  = DualEnsemble("wbf", 0.7, 0.1, classes=["Trophozoite", "WBC"], weights = [1,1,1,1], wbf_reduction="mean")
    detr_ens = DualEnsemble("wbf", 0.7, 0.1, classes=["Trophozoite", "WBC"], weights = [1,1,1,1], wbf_reduction="mean")
    all_ens = DualEnsemble("wbf", 0.6, 0.1,classes=["Trophozoite", "WBC"], weights = [1,1], wbf_reduction="mean")

    yolo_df = yolo_ens(yolo_tta_dfs)
    detr_df = detr_ens(detr_tta_dfs)
    all_df = all_ens([yolo_df, detr_df])

    # Add NEG predictions to the final submission
    os.makedirs('submissions', exist_ok=True)
    neg_all_df = pd.read_csv("data/csv_files/NEG_OR_NOT2.csv")
    final_submission_df = add_negs_to_submission(df=all_df, neg_csv="data/csv_files/NEG_OR_NOT2.csv", test_csv="data/csv_files/Test.csv")
    final_submission_df.to_csv("submissions/final_submission_ensemble_wbf.csv", index=False)
    print(f"File is saved to: submissions/final_submission_ensemble_wbf.csv")

def full_pipeline(tta_files, config_name):
    config_file = f"parameters/postprocessing_config_files/{config_name}.yaml"
    config = load_yaml_config(config_file)
    param_config = create_structured_config(config["parameters"])
    print(param_config)
    detr_tta_files = ["data/predictions/"+f for f in tta_files if "det" in f and "neg" not in f]
    yolo_tta_files = ["data/predictions/"+f for f in tta_files if "yol" in f]

    print(detr_tta_files)
    print(yolo_tta_files)
    all_df = run_postprocessing(param_config, 1, yolo_tta_files, detr_tta_files)
     # Add NEG predictions to the final submission
    os.makedirs('submissions', exist_ok=True)
    neg_all_df = pd.read_csv("data/csv_files/NEG_OR_NOT2.csv")
    final_submission_df = add_negs_to_submission(df=all_df, neg_csv="data/csv_files/NEG_OR_NOT2.csv", test_csv="data/csv_files/Test.csv")
    final_submission_df.to_csv(f"submissions/final_submission_{config_name}.csv", index=False)
    print(f"File is saved to: submissions/final_submission_{config_name}.csv")




if __name__ == "__main__":
    tta_files = os.listdir("data/tta_predicitions_ready_for_ensemble")
    print(tta_files)


    ensemble_wbf_only(tta_files=tta_files)
    full_pipeline(tta_files,"exalted_sweep" )
    #  --- POSTPROCESSING ---
    # config_file = "parameters/postprocessing_config_files/exalted_sweep.yaml"
    # config = load_yaml_config(config_file)
    # param_config = create_structured_config(config["parameters"])
    # print(param_config)
    # print(final_preds.keys())
    # detr_tta_files = ["data/predictions/"+f for f in tta_files if "det" in f]
    # yolo_tta_files = ["data/predictions/"+f for f in tta_files if "yol" in f]

    # print(detr_tta_files)
    # print(yolo_tta_files)
    # all_df = run_postprocessing(param_config, 1, yolo_tta_files, detr_tta_files)


    # detr_tta_files = [pd.read_csv("data/predictions/"+f) for f in tta_files if "det" in f]
    # yolo_tta_files = [pd.read_csv("data/predictions/"+f)for f in tta_files if "yol" in f]

    # yolo_ens  = DualEnsemble("wbf", 0.7, 0.1, classes=["Trophozoite", "WBC"], weights = [1,1,1,1], wbf_reduction="mean")
    # detr_ens = DualEnsemble("wbf", 0.7, 0.1, classes=["Trophozoite", "WBC"], weights = [1,1,1,1], wbf_reduction="mean")
    # all_ens = DualEnsemble("wbf", 0.6, 0.1,classes=["Trophozoite", "WBC"], weights = [1,1], wbf_reduction="mean")

    # yolo_df = yolo_ens(yolo_tta_files)
    # detr_ens = detr_ens(detr_tta_files)
    # all_df = all_ens([yolo_df, detr_ens])

    # # Add NEG predictions to the final submission
    # os.makedirs('submissions', exist_ok=True)
    # neg_all_df = pd.read_csv("data/csv_files/NEG_OR_NOT2.csv")
    # final_submission_df = add_negs_to_submission(df=all_df, neg_csv="data/csv_files/NEG_OR_NOT2.csv", test_csv="data/csv_files/Test.csv")
    # final_submission_df.to_csv("submissions/final_submission_exalted_sweep.csv", index=False)

