import os
import argparse
import numpy as np
import pandas as pd
import wandb
import yaml
from pathlib import Path
from postprocessing.postprocess import postprocessing_pipeline, ensemble_class_specific_pipeline, apply_confidence_threshold
from util.wbf import weighted_boxes_fusion_df
from util.mAP_zindi import mAP_zindi_calculation
from util.ensemble import DualEnsemble, Ensemble
import concurrent.futures
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def score_on_validation_set(df, fold_num, split_csv, train_csv):
    logging.info(f"Scoring on validation set for fold {fold_num}")
    df = df.copy()
    folds_df = pd.read_csv(split_csv)
    train_df = pd.read_csv(train_csv)

    fold_df = folds_df[folds_df['Split'] == fold_num]
    val_imgs = set(fold_df['Image_ID'])

    pred_df = df[df['Image_ID'].isin(val_imgs)]
    df_val = train_df[train_df['Image_ID'].isin(val_imgs)]
    logging.info(f"Number of images in validation set: {len(val_imgs)}")
    logging.info(f"Number of predictions in prediction set: {len(pred_df)}")
    
    map_score, ap_dict, lamr_dict = mAP_zindi_calculation(df_val, pred_df)
    logging.info(f"Completed scoring for fold {fold_num}")
   
    return map_score, ap_dict, lamr_dict


def load_yaml_config(filepath):
    logging.info(f"Loading YAML config from {filepath}")
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    logging.info(f"YAML config loaded successfully from {filepath}")
    return config


def create_structured_config(wandb_config):
    logging.info("Creating structured config from wandb parameters")
    config = {
        'input': {},
        'postprocessing': {
            'individual_detr': {},
            'individual_yolo11': {},
            'individual_yolo9': {},
            'ensemble_yolo': {},
            'ensemble_all': {}
        },
        'troph_weights': {},
        'wbc_weights': {}
    }
    
    # Map wandb parameters to config structure
    for key, value in wandb_config.items():
        if key.startswith('input_'):
            config['input'][key[6:]] = value  # remove 'input_' prefix
        elif key.startswith('postprocessing_'):
            parts = key.split('_')[1:]  # Remove 'postprocessing_' prefix
            
            if len(parts) >= 3:
                section = '_'.join(parts[:2])
                param = '_'.join(parts[2:])
                
                # Handle class-specific parameters
                if param.startswith('troph_') or param.startswith('wbc_'):
                    class_type, param_name = param.split('_', 1)
                    if section in config['postprocessing']:
                        if f'{class_type}_params' not in config['postprocessing'][section]:
                            config['postprocessing'][section][f'{class_type}_params'] = {}
                        config['postprocessing'][section][f'{class_type}_params'][param_name] = value
                else:
                    if section in config['postprocessing']:
                        config['postprocessing'][section][param] = value
        
        # Handle ensemble weights
        elif any(key.startswith(prefix) for prefix in ['detr_weight_', 'yolo_weight_', 'yolo11_weight_', 'yolo9_weight_']):
            weight_parts = key.split('_')
            model = weight_parts[0]
            class_type = weight_parts[-1]
            weight_key = f"{model}_weight"
            # Assign weights based on class_type
            if class_type.lower() == 'trophozoite':
                config['troph_weights'][weight_key] = value
            elif class_type.lower() == 'wbc':
                config['wbc_weights'][weight_key] = value
        
        # Map output path
        elif key == 'output_path':
            config['output_path'] = value
    
    logging.info("Structured config created successfully")
    return config


def run_fold(config, fold_num, yolo11_cv_files, yolo9_cv_files, detr_cv_files):
    logging.info(f"Running fold {fold_num}")

    def create_pipeline_config(stage_config, base_paths):
        pipeline_config = {
            'DATA_DIR': base_paths['DATA_DIR'],
            'NEG_CSV': base_paths['NEG_CSV'],
            'TEST_CSV': base_paths['TEST_CSV'],
            'TRAIN_CSV': base_paths['TRAIN_CSV'],
            'SPLIT_CSV': base_paths['SPLIT_CSV'],
            'fold_num': fold_num,
        }

        # Handle class-specific parameters
        if 'troph_params' in stage_config:
            for key, value in stage_config['troph_params'].items():
                pipeline_config[f'troph_{key}'] = value
        if 'wbc_params' in stage_config:
            for key, value in stage_config['wbc_params'].items():
                pipeline_config[f'wbc_{key}'] = value

        # Add any non-class-specific parameters
        for key, value in stage_config.items():
            if key not in ['troph_params', 'wbc_params']:
                pipeline_config[key] = value

        return pipeline_config

    try:
        # Process YOLO predictions
        yolo_dfs = []
    
        # Process YOLO11
        logging.info(f"Processing YOLO11 predictions for fold {fold_num}")
        yolo_df = pd.read_csv(os.path.join(config['input']['yolo11_csv_dir'], yolo11_cv_files[fold_num - 1]))
        yolo_pipeline_config = create_pipeline_config(
            config['postprocessing']['individual_yolo11'],
            config['input']
        )
        yolo_df = postprocessing_pipeline(yolo_pipeline_config, yolo_df)
        yolo_dfs.append(yolo_df)
        logging.info(f"Completed YOLO11 processing for fold {fold_num}")

        # Process YOLO9 if available
        if 'yolo9_csv_dir' in config['input']:
            logging.info(f"Processing YOLO9 predictions for fold {fold_num}")
            yolo_df = pd.read_csv(os.path.join(config['input']['yolo9_csv_dir'], yolo9_cv_files[fold_num - 1]))
            yolo_df = apply_confidence_threshold(yolo_df, 0.1)
            yolo_pipeline_config = create_pipeline_config(
                config['postprocessing']['individual_yolo9'],
                config['input']
            )
            yolo_df = postprocessing_pipeline(yolo_pipeline_config, yolo_df)
            
            yolo_dfs.append(yolo_df)
            logging.info(f"Completed YOLO9 processing for fold {fold_num}")

        # Ensemble YOLO predictions if multiple files
        if len(yolo_dfs) > 1:
            logging.info(f"Ensembling YOLO predictions for fold {fold_num}")
            yolo_ensemble_config = create_pipeline_config(
                config['postprocessing']['ensemble_yolo'],
                config['input']
            )
            yolo_weights = [[config['troph_weights']['yolo11_weight'], config['troph_weights']['yolo9_weight']],
                            [config['wbc_weights']['yolo11_weight'], config['wbc_weights']['yolo9_weight']]]
            yolo_df_all = ensemble_class_specific_pipeline(
                CONFIG=yolo_ensemble_config,
                df_list=yolo_dfs,
                weight_list=yolo_weights
            )
            logging.info(f"Completed YOLO ensembling for fold {fold_num}")
        else:
            yolo_df_all = yolo_dfs[0]

        # Process DETR predictions
        logging.info(f"Processing DETR predictions for fold {fold_num}")
        detr_file = os.path.join(config['input']['detr_csv_dir'], detr_cv_files[fold_num - 1])
        detr_df = pd.read_csv(detr_file)
        detr_pipeline_config = create_pipeline_config(
            config['postprocessing']['individual_detr'],
            config['input']
        )
        detr_df = postprocessing_pipeline(detr_pipeline_config, detr_df)
        logging.info(f"Completed DETR processing for fold {fold_num}")

        # Final ensemble
        logging.info(f"Running final ensemble for fold {fold_num}")
        final_pipeline_config = create_pipeline_config(
            config['postprocessing']['ensemble_all'],
            config['input']
        )
        final_weights = [[config['troph_weights']['yolo_weight'], config['troph_weights']['detr_weight']],
                         [config['wbc_weights']['yolo_weight'], config['wbc_weights']['detr_weight']]]
        all_df = ensemble_class_specific_pipeline(
            CONFIG=final_pipeline_config,
            df_list=[yolo_df_all, detr_df],
            weight_list=final_weights
        )
        logging.info(f"Completed final ensemble for fold {fold_num}")

        # Calculate metrics
        logging.info(f"Calculating metrics for fold {fold_num}")
        map_score, ap_dict, lamr_dict = score_on_validation_set(
            df=all_df,
            fold_num=fold_num,
            split_csv=config['input']['SPLIT_CSV'],
            train_csv=config['input']['TRAIN_CSV']
        )
        logging.info(f"Metrics calculated for fold {fold_num}")

        return {
            "fold_num": fold_num,
            "mAP": map_score,
            "AP_troph": ap_dict['Trophozoite'],
            "AP_WBC": ap_dict['WBC'],
            "lamr_troph": lamr_dict['Trophozoite'],
            "lamr_WBC": lamr_dict['WBC']
        }

    except Exception as e:
        logging.error(f"Error occurred during fold {fold_num}: {e}")
        raise


def run_experiment(config_file):
    # Initialize wandb
    with wandb.init() as run:
        # Load base config and update with wandb parameters
        logging.info(f"Running experiment with config file: {config_file}")
        base_config = load_yaml_config(config_file)
        config = create_structured_config(wandb.config)

        # Cross-validation files
        detr_cv_files = ['fold_1.csv', 'fold_2.csv', 'fold_3.csv', 'fold_4.csv', 'fold_5.csv']
        yolo11_cv_files = ['77_val.csv', '79_val.csv', '81_val.csv', '84_val.csv', '87_val.csv']
        yolo9_cv_files = ['fold_1.csv', 'fold_2.csv', 'fold_3.csv', 'fold_4.csv', 'fold_5.csv']

        # Run cross-validation folds in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_fold, config, i, yolo11_cv_files, yolo9_cv_files, detr_cv_files) for i in range(1, 6)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Track metrics
        cv_metrics = {
            "mAP": [],
            "AP_troph": [],
            "AP_WBC": [],
            "lamr_troph": [],
            "lamr_WBC": []
        }

        # Collect metrics from results
        for result in results:
            fold_num = result["fold_num"]
            logging.info(f"Fold {fold_num} metrics: mAP: {result['mAP']}, AP_Troph: {result['AP_troph']}, AP_WBC: {result['AP_WBC']}, LAMR_Troph: {result['lamr_troph']}, LAMR_WBC: {result['lamr_WBC']}")
            
            cv_metrics["mAP"].append(result["mAP"])
            cv_metrics["AP_troph"].append(result["AP_troph"])
            cv_metrics["AP_WBC"].append(result["AP_WBC"])
            cv_metrics["lamr_troph"].append(result["lamr_troph"])
            cv_metrics["lamr_WBC"].append(result["lamr_WBC"])

        # Calculate and log mean metrics
        mean_metrics = {k: np.mean(v) for k, v in cv_metrics.items()}
        logging.info(f"Mean metrics: {mean_metrics}")
        wandb.log(mean_metrics)

        # Save best configuration
        if mean_metrics['mAP'] > wandb.run.summary.get('best_AP_mean', 0):
            wandb.run.summary['best_AP_mean'] = mean_metrics['mAP']
            best_config_path = os.path.join(wandb.run.dir, 'best_config.yaml')
            with open(best_config_path, 'w') as f:
                yaml.dump(config, f)
            logging.info(f"Best configuration saved to {best_config_path}")


def run_multiple_experiments(config_files):
    # Run multiple experiments in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_experiment, config_file) for config_file in config_files]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    config_files = ["parameters/postprocessing_config_files/class_specific_sweep.yaml"]  # Add more config files as needed
    run_multiple_experiments(config_files)