import os
import argparse
import numpy as np
import pandas as pd
import wandb
import yaml
from pathlib import Path
from postprocessing.postprocess import postprocessing_pipeline, ensemble_pipeline
from util.wbf import weighted_boxes_fusion_df
from util.mAP_zindi import mAP_zindi_calculation
from util.ensemble import DualEnsemble, Ensemble

def score_on_validation_set(df, fold_num, split_csv, train_csv):
    df = df.copy()
    folds_df = pd.read_csv(split_csv)
    train_df = pd.read_csv(train_csv)

    fold_df = folds_df[folds_df['Split'] == fold_num]
    val_imgs = set(fold_df['Image_ID'])

    pred_df = df[df['Image_ID'].isin(val_imgs)]
    df_val = train_df[train_df['Image_ID'].isin(val_imgs)]
    print(f"Number of images in validation set: {len(val_imgs)}")
    print(f"Number of predictions in prediction set: {len(pred_df)}")
    map_score, ap_dict, lamr_dict = mAP_zindi_calculation(df_val, pred_df)
   
    return map_score, ap_dict, lamr_dict


def load_yaml_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_structured_config(wandb_config):
    """Creates a structured configuration from wandb parameters. The base structure can be found in the """
    config = {
        'input': {},
        'postprocessing': {
            'individual_detr': {},
            'individual_yolo11': {},
            'individual_yolo9': {},
            'ensemble_yolo': {},
            'ensemble_all': {}
        }
    }
    
    # Map wandb parameters to config structure
    for key, value in wandb_config.items():
        # print(key, value)
        if key.startswith('input_'):
            config['input'][key[6:]] = value #remove 'input_' prefix
        elif key.startswith('postprocessing_'):
            parts = key.split('_')[1:]  # Remove 'postprocessing_' prefix
            # print(parts)
            if len(parts) >= 3:
                section = '_'.join(parts[:2])
                param = '_'.join(parts[2:])
                # print(section, param)
                if section in config['postprocessing']:
                    config['postprocessing'][section][param] = value
        elif key in ['detr_weight', 'yolo_weight', "yolo11_weight", "yolo9_weight"]:
            config[key] = value
        elif key == 'output_path':
            config['output_path'] = value
            
    return config

def run_experiment(config_file):
    """Main experiment function with wandb integration."""
    # Initialize wandb
    with wandb.init() as run:
        # Load base config and update with wandb parameters
        base_config = load_yaml_config(config_file)
        config = create_structured_config(wandb.config)
        print(config,"/n/n")
        print(base_config)
        # Cross-validation files
        detr_cv_files = ['fold_1.csv', 'fold_2.csv', 'fold_3.csv', 'fold_4.csv', 'fold_5.csv']
        yolo11_cv_files = ['77_val.csv', '79_val.csv', '81_val.csv', '84_val.csv', '87_val.csv']
        yolo9_cv_files = ['fold_1.csv', 'fold_2.csv', 'fold_3.csv', 'fold_4.csv', 'fold_5.csv']
        # Track metrics
        cv_metrics = {
            "mAP": [], 
            "AP_troph": [], 
            "AP_WBC": [], 
            "lamr_troph": [], 
            "lamr_WBC": []
        }
        
        # Run cross-validation folds
        for i in range(1, 6):
            print(f"Running fold {i}")
            
            # Create pipeline config for each stage
            def create_pipeline_config(stage_config, base_paths):
                pipeline_config = {
                    'DATA_DIR': base_paths['DATA_DIR'],
                    'NEG_CSV': base_paths['NEG_CSV'],
                    'TEST_CSV': base_paths['TEST_CSV'],
                    'TRAIN_CSV': base_paths['TRAIN_CSV'],
                    'SPLIT_CSV': base_paths['SPLIT_CSV'],
                    'fold_num': i,
                }
                
                # Map all parameters from stage_config to pipeline_config
                for key, value in stage_config.items():
                    pipeline_config[key] = value
                
                return pipeline_config

            # Process YOLO predictions
            yolo_dfs = []
           
            yolo_df = pd.read_csv(os.path.join(config['input']['yolo11_csv_dir'], yolo11_cv_files[i - 1]))
            yolo_pipeline_config = create_pipeline_config(
                config['postprocessing']['individual_yolo11'],
                config['input']
            )
            yolo_df = postprocessing_pipeline(yolo_pipeline_config, yolo_df)
            # Apply YOLO weight to confidence scores
            
            yolo_dfs.append(yolo_df)
            #CHANGE to specific yolo9 config
            if 'yolo9_csv_dir' in config['input']:
                yolo_df = pd.read_csv(os.path.join(config['input']['yolo9_csv_dir'], yolo9_cv_files[i - 1]))
                yolo_pipeline_config = create_pipeline_config(
                    config['postprocessing']['individual_yolo9'],
                    config['input']
                )
                yolo_df = postprocessing_pipeline(yolo_pipeline_config, yolo_df)
                # Apply YOLO weight to confidence scores
                yolo_dfs.append(yolo_df)
            
            # Ensemble YOLO predictions if multiple files
            if len(yolo_dfs) > 1:
                yolo_ensemble_config = create_pipeline_config(
                    config['postprocessing']['ensemble_yolo'],
                    config['input']
                )
                yolo_df_all = ensemble_pipeline(CONFIG=yolo_ensemble_config, df_list=yolo_dfs, weight_list=[config["yolo11_weight"],config["yolo9_weight"]])
                
            else:
                yolo_df_all = yolo_dfs[0]
            
            # Process DETR predictions
            detr_file = os.path.join(config['input']['detr_csv_dir'], detr_cv_files[i - 1])
            detr_df = pd.read_csv(detr_file)
            detr_pipeline_config = create_pipeline_config(
                config['postprocessing']['individual_detr'],
                config['input']
            )
            detr_df = postprocessing_pipeline(detr_pipeline_config, detr_df)
            # Apply DETR weight to confidence scores

            
            # Final ensemble
            
            final_pipeline_config = create_pipeline_config(
                config['postprocessing']['ensemble_all'],
                config['input']
            )
            all_df = ensemble_pipeline(CONFIG=final_pipeline_config, df_list=[yolo_df_all, detr_df], weight_list=[config['yolo_weight'], config['detr_weight']])

            # Calculate metrics
            map_score, ap_dict, lamr_dict = score_on_validation_set(
                df=all_df,
                fold_num=i,
                split_csv=config['input']['SPLIT_CSV'],
                train_csv=config['input']['TRAIN_CSV']
            )
            
          
            # Calculate and log mean metrics
            mean_metrics = {k: np.mean(v) for k, v in cv_metrics.items()}
            wandb.log(mean_metrics)
            
            # Save best configuration
            if mean_metrics['mAP'] > wandb.run.summary.get('best_AP_mean', 0):
                wandb.run.summary['best_AP_mean'] = mean_metrics['mAP']
                best_config_path = os.path.join(wandb.run.dir, 'best_config.yaml')
                with open(best_config_path, 'w') as f:
                    yaml.dump(config, f)
            # Early stopping check on fold 1
            if i == 1 and map_score < 0.86:
                print(f"Early stopping: Fold 1 mAP ({map_score:.4f}) below threshold")
                wandb.log({
                    "mAP": map_score,
                    "AP_troph": ap_dict['Trophozoite'],
                    "AP_WBC": ap_dict['WBC'],
                    "lamr_troph": lamr_dict['Trophozoite'],
                    "lamr_WBC": lamr_dict['WBC'],
                    "early_stopped": True,
                    "completed_folds": 1
                })
                return  # Exit this trial and move to next wandb suggestion
            
            # Store metrics
            print(f"Fold {i} metrics:")
            print(f"mAP: {map_score}")
            print(f"AP: {ap_dict}")
            print(f"LAMR: {lamr_dict}")
            cv_metrics["mAP"].append(map_score)
            cv_metrics["AP_troph"].append(ap_dict['Trophozoite'])
            cv_metrics["AP_WBC"].append(ap_dict['WBC'])
            cv_metrics["lamr_troph"].append(lamr_dict['Trophozoite'])
            cv_metrics["lamr_WBC"].append(lamr_dict['WBC'])
        
if __name__ == "__main__":
    config = "parameters/postprocessing_config_files/sweep.yaml"
    run_experiment(config)