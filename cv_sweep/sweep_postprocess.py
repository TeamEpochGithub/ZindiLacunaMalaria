import glob
import os
from contextlib import nullcontext

import numpy as np
import pandas as pd
import wandb
import yaml
from scipy.stats import describe
from tqdm import tqdm

from util.save import save_with_negs
from util.mAP_zindi import mAP_zindi_calculation
from postprocessing.postprocess_functions import (
    postprocessing_pipeline,
    ensemble_class_specific_pipeline,
)
import concurrent.futures
import logging
from torchvision.ops import nms
import torch

def _apply_nms_to_boxes(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, 
                       iou_threshold: float) -> tuple:
    """Apply NMS to boxes, handling all classes together."""
    # Apply NMS across all classes at once
    keep = nms(boxes, scores, iou_threshold)

    return keep

def apply_nms_to_df(df, iou_threshold, score_col="confidence"):
    df = df.copy()
    boxes = df[["xmin","ymin","xmax", "ymax"]].values
    scores = df[score_col].values
    labels = df["class"].map({"Trophozoite": 0, "WBC": 1}).to_numpy()
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    labels = torch.tensor(labels)
    keep = _apply_nms_to_boxes(boxes, scores, labels, iou_threshold)
    df = df.iloc[keep]
    return df

def score_on_validation_set(df, fold_num, split_csv, train_csv):
    df = df.copy()
    folds_df = pd.read_csv(split_csv)
    train_df = pd.read_csv(train_csv)

    fold_df = folds_df[folds_df["Split"] == fold_num]
    val_imgs = set(fold_df["Image_ID"])

    pred_df = df[df["Image_ID"].isin(val_imgs)]
    df_val = train_df[train_df["Image_ID"].isin(val_imgs)]
    print(f"Number of images in validation set: {len(val_imgs)}")
    print(f"Number of predictions in prediction set: {len(pred_df)}")
    map_score, ap_dict, lamr_dict = mAP_zindi_calculation(df_val, pred_df)

    return map_score, ap_dict, lamr_dict


def load_yaml_config(filepath):
    logging.info(f"Loading YAML config from {filepath}")
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
    logging.info(f"YAML config loaded successfully from {filepath}")
    return config


def create_structured_config(wandb_config):
    logging.info("Creating structured config from wandb parameters")
    config = {
        "input": {},
        "postprocessing": {
            "ensemble_ttayolo": {},
            "ensemble_ttadetr": {},
            "individual_detr": {},
            "individual_yolo11": {},
            "ensemble_all": {},
        }
    }

    # Map wandb parameters to config structure
    for key, value in wandb_config.items():
        if key.startswith("input_"):
            config["input"][key[6:]] = value  # remove 'input_' prefix
        elif key.startswith("postprocessing_"):
            parts = key.split("_")[1:]  # Remove 'postprocessing_' prefix

            if len(parts) >= 3:
                section = "_".join(parts[:2])
                param = "_".join(parts[2:])

                # Handle class-specific parameters
                if 'Trophozoite' in param or 'WBC' in param:
                    class_type, param_name = param.split("_", 1)
                    if section in config["postprocessing"]:
                        if f"{class_type}_params" not in config["postprocessing"][section]:
                            config["postprocessing"][section][f"{class_type}_params"] = {}

                        config["postprocessing"][section][f"{class_type}_params"][param_name] = value
                else:
                    if section in config["postprocessing"]:
                        config["postprocessing"][section][param] = value

        
        # Map output path
        elif key == "output_path":
            config["output_path"] = value

    logging.info("Structured config created successfully")
    return config


def run_fold(config, fold_num, yolo11_cv_files_split, detr_cv_files):
    logging.info(f"Running fold {fold_num}")

    def create_pipeline_config(stage_config, base_paths):
        print(base_paths)
        pipeline_config = {
            "DATA_DIR": base_paths["DATA_DIR"],
            "NEG_CSV": base_paths["NEG_CSV"],
            "TEST_CSV": base_paths["TEST_CSV"],
            "TRAIN_CSV": base_paths["TRAIN_CSV"],
            "SPLIT_CSV": base_paths["SPLIT_CSV"],
            "fold_num": fold_num,
        }

        # Handle class-specific parameters
        if "Trophozoite_params" in stage_config:
            for key, value in stage_config["Trophozoite_params"].items():
                pipeline_config[f"Trophozoite_{key}"] = value
        if "WBC_params" in stage_config:
            for key, value in stage_config["WBC_params"].items():
                pipeline_config[f"WBC_{key}"] = value

        # Add any non-class-specific parameters
        for key, value in stage_config.items():
            if key not in ["Trophozoite_params", "WBC_params"]:
                pipeline_config[key] = value

        return pipeline_config

    # Process YOLO predictions
    yolo_dfs = []
    # Process YOLO11
    logging.info(
        f"Processing YOLO11 predictions for fold {fold_num}. Will run TTA ensemble."
    )
    for tta_flip in range(len(yolo11_cv_files_split)):
        yolo_df = pd.read_csv(yolo11_cv_files_split[tta_flip])
        yolo_dfs.append(yolo_df)
    yolo_tta_config = create_pipeline_config(
        config["postprocessing"]["ensemble_ttayolo"], config["input"]
    )

    yolo_tta_df = ensemble_class_specific_pipeline(
        CONFIG=yolo_tta_config,
        df_list=yolo_dfs,
        weight_list=[[1, 1, 1, 1], [1, 1, 1, 1]],
    )  # weight list expects 4 weights for troph and wbc
    logging.info(f"Completed YOLO11 ensembling of TTA predictions for fold {fold_num}")
    yolo_individual_config = create_pipeline_config(
        config["postprocessing"]["individual_yolo11"], config["input"]
    )
    yolo_tta_df = postprocessing_pipeline(yolo_individual_config, yolo_tta_df)
    logging.info(f"Completed YOLO11 postprocessing for fold {fold_num}")

    # Process DETR predictions
    detr_dfs = []
    for tta_flip in range(len(detr_cv_files)):
        detr_df = pd.read_csv(detr_cv_files[tta_flip])
        detr_dfs.append(detr_df)

    detr_tta_config = create_pipeline_config(
        config["postprocessing"]["ensemble_ttadetr"], config["input"]
    )
    detr_tta_df = ensemble_class_specific_pipeline(
        CONFIG=detr_tta_config,
        df_list=detr_dfs,
        weight_list=[[1, 1, 1, 1], [1, 1, 1, 1]],
    )  # weight list expects 4 weights for troph and wbc
    logging.info(f"Completed DETR ensembling of TTA predictions for fold {fold_num}")

    detr_pipeline_config = create_pipeline_config(
        config["postprocessing"]["individual_detr"], config["input"]
    )
    print(detr_pipeline_config)
    if detr_pipeline_config["allclass_use_class_independent_nms"]:
        detr_tta_df = apply_nms_to_df(detr_tta_df, detr_pipeline_config["allclass_nms_iou_threshold"])

    detr_df = postprocessing_pipeline(detr_pipeline_config, detr_tta_df)

    # Final ensemble
    logging.info(f"Running final ensemble for fold {fold_num}")
    final_pipeline_config = create_pipeline_config(
        config["postprocessing"]["ensemble_all"], config["input"]
    )
    final_weights = [[1,1], [1,1]]

    all_df = ensemble_class_specific_pipeline(
        CONFIG=final_pipeline_config,
        df_list=[yolo_tta_df, detr_df],
        weight_list=final_weights,
    )
    all_df = postprocessing_pipeline(final_pipeline_config, all_df)
    # Calculate metrics
    map_score, ap_dict, lamr_dict = score_on_validation_set(
        df=all_df,
        fold_num=fold_num,
        split_csv=config["input"]["SPLIT_CSV"],
        train_csv=config["input"]["TRAIN_CSV"],
    )
    logging.info(f"Completed final ensemble for fold {fold_num}")

    # Store metrics
    print(f"Fold {fold_num} metrics:")
    print(f"mAP: {map_score}")
    print(f"AP: {ap_dict}")
    print(f"LAMR: {lamr_dict}")

    # Early stopping check on fold 1
    if fold_num == 1 and map_score < 0.86:
        print(f"Early stopping: Fold 1 mAP ({map_score:.4f}) below threshold")
        return None  # Exit this trial and move to next wandb suggestion

    return {
        "mAP": map_score,
        "AP_troph": ap_dict["Trophozoite"],
        "AP_WBC": ap_dict["WBC"],
        "lamr_troph": lamr_dict["Trophozoite"],
        "lamr_WBC": lamr_dict["WBC"],
    }


def run_experiment(config_file):
    # Initialize wandb
    with wandb.init(project="final_pp_sweep") as run:
        # Load base config and update with wandb parameters
        logging.info(f"Running experiment with config file: {config_file}")
        base_config = load_yaml_config(config_file)
        config = create_structured_config(wandb.config)
        # Update wandb config
        wandb.config.update(config)

        print(config)
        # best_config_path = os.path.join(wandb.run.dir, "best_config.yaml")
        # with open(best_config_path, "w") as f:
            # yaml.dump(config, f)
        # Cross-validation files
        detr_cv_files = [
            [
                "data/predictions/SPLIT1/detr/fold1_tta1.csv",
                "data/predictions/SPLIT1/detr/fold1_tta2.csv",
                "data/predictions/SPLIT1/detr/fold1_tta3.csv",
                "data/predictions/SPLIT1/detr/fold1_tta4.csv",
            ],
            [
                "data/predictions/SPLIT2/detr/fold2_tta1.csv",
                "data/predictions/SPLIT2/detr/fold2_tta2.csv",
                "data/predictions/SPLIT2/detr/fold2_tta3.csv",
                "data/predictions/SPLIT2/detr/fold2_tta4.csv",
            ],
            [
                "data/predictions/SPLIT3/detr/fold3_tta1.csv",
                "data/predictions/SPLIT3/detr/fold3_tta2.csv",
                "data/predictions/SPLIT3/detr/fold3_tta3.csv",
                "data/predictions/SPLIT3/detr/fold3_tta4.csv",
            ],
            [
                "data/predictions/SPLIT4/detr/fold4_tta1.csv",
                "data/predictions/SPLIT4/detr/fold4_tta2.csv",
                "data/predictions/SPLIT4/detr/fold4_tta3.csv",
                "data/predictions/SPLIT4/detr/fold4_tta4.csv",
            ],
            [
                "data/predictions/SPLIT5/detr/fold5_tta1.csv",
                "data/predictions/SPLIT5/detr/fold5_tta2.csv",
                "data/predictions/SPLIT5/detr/fold5_tta3.csv",
                "data/predictions/SPLIT5/detr/fold5_tta4.csv",
            ],
        ]
        yolo11_cv_files = [
            [
                "data/predictions/SPLIT1/yolo_models/predictions_0.csv",
                "data/predictions/SPLIT1/yolo_models/predictions_1.csv",
                "data/predictions/SPLIT1/yolo_models/predictions_2.csv",
                "data/predictions/SPLIT1/yolo_models/predictions_3.csv",
            ],
            [
                "data/predictions/SPLIT2/yolo_models/predictions_0.csv",
                "data/predictions/SPLIT2/yolo_models/predictions_1.csv",
                "data/predictions/SPLIT2/yolo_models/predictions_2.csv",
                "data/predictions/SPLIT2/yolo_models/predictions_3.csv",
            ],
            [
                "data/predictions/SPLIT3/yolo_models/predictions_0.csv",
                "data/predictions/SPLIT3/yolo_models/predictions_1.csv",
                "data/predictions/SPLIT3/yolo_models/predictions_2.csv",
                "data/predictions/SPLIT3/yolo_models/predictions_3.csv",
            ],
            [
                "data/predictions/SPLIT4/yolo_models/predictions_0.csv",
                "data/predictions/SPLIT4/yolo_models/predictions_1.csv",
                "data/predictions/SPLIT4/yolo_models/predictions_2.csv",
                "data/predictions/SPLIT4/yolo_models/predictions_3.csv",
            ],
            [
                "data/predictions/SPLIT5/yolo_models/predictions_0.csv",
                "data/predictions/SPLIT5/yolo_models/predictions_1.csv",
                "data/predictions/SPLIT5/yolo_models/predictions_2.csv",
                "data/predictions/SPLIT5/yolo_models/predictions_3.csv",
            ],
        ]

        results = []
        for i in range(1, 6):
            result = run_fold(config, i, yolo11_cv_files[i - 1], detr_cv_files[i - 1])
            if result is None:
                break
            results.append(result)
            # Log fold-wise metrics
            wandb.log(
                {
                    f"fold_{i}_mAP": result["mAP"],
                    f"fold_{i}_AP_troph": result["AP_troph"],
                    f"fold_{i}_AP_WBC": result["AP_WBC"],
                    f"fold_{i}_lamr_troph": result["lamr_troph"],
                    f"fold_{i}_lamr_WBC": result["lamr_WBC"],
                }
            )

        # Check if any results were collected
        if results:
            # Initialize cv_metrics
            cv_metrics = {k: [] for k in results[0].keys()}

            # Collect metrics from results
            for result in results:
                print(result)
                logging.info(
                    f"metrics: mAP: {result['mAP']}, AP_Troph: {result['AP_troph']}, AP_WBC: {result['AP_WBC']}, LAMR_Troph: {result['lamr_troph']}, LAMR_WBC: {result['lamr_WBC']}"
                )

                for k in result:
                    cv_metrics[k].append(result[k])

            # Calculate and log mean metrics
            mean_metrics = {f"mean_{k}": np.mean(v) for k, v in cv_metrics.items()}
            logging.info(f"Mean metrics: {mean_metrics}")
            wandb.log(mean_metrics)

            # Save best configuration
            if mean_metrics["mean_mAP"] > wandb.run.summary.get("best_AP_mean", 0):
                wandb.run.summary["best_AP_mean"] = mean_metrics["mean_mAP"]
                best_config_path = os.path.join(wandb.run.dir, "best_config.yaml")
                with open(best_config_path, "w") as f:
                    yaml.dump(config, f)
                logging.info(f"Best configuration saved to {best_config_path}")
        else:
            logging.info("No results to aggregate.")


if __name__ == "__main__":
    config_file = "parameters/postprocessing_config_files/save.yaml"
    run_experiment(config_file)
