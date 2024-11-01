import os
import yaml
import pandas as pd
from pathlib import Path
from postprocessing.postprocessing import postprocessing_pipeline
from util.wbf import weighted_boxes_fusion_df

def load_config(config_path):
    """Load and validate the ensemble configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Resolve path variables
    def resolve_paths(cfg, base_cfg):
        if isinstance(cfg, dict):
            for key, value in cfg.items():
                if isinstance(value, str) and value.startswith("${"):
                    # Extract the path from ${path.to.value}
                    path = value[2:-1].split('.')
                    result = base_cfg
                    for p in path:
                        result = result[p]
                    cfg[key] = result
                elif isinstance(value, (dict, list)):
                    resolve_paths(value, base_cfg)
        elif isinstance(cfg, list):
            for item in cfg:
                if isinstance(item, (dict, list)):
                    resolve_paths(item, base_cfg)
    
    resolve_paths(config, config)
    return config

def process_dataframes(csv_files, pp_config, base_config):
    """Process multiple dataframes with optional postprocessing."""
    if not csv_files:
        return pd.DataFrame()
    
    if isinstance(csv_files, str):
        csv_files = [csv_files]
        
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if pp_config.get('enabled', False):
            # Merge base paths with postprocessing config
            full_config = pp_config['config']
            df = postprocessing_pipeline(full_config, df)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def ensemble_pipeline(config_path):
    """
    Run the ensemble pipeline using configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Validate YOLO models
    yolo_pattern = config['input']['yolo_models_path']
    n_yolo_models = len(list(Path().glob(yolo_pattern)))
    n_yolo_csvs = len(config['input']['yolo_csv_paths'])
    
    if n_yolo_models != n_yolo_csvs:
        raise ValueError(f"Number of YOLO models ({n_yolo_models}) and CSV files ({n_yolo_csvs}) do not match")
    
    # Process YOLO models
    yolo_df = process_dataframes(
        config['input']['yolo_csv_paths'],
        config['postprocessing']['individual_yolo'],
        config
    )
    
    # Process DETR model
    detr_df = process_dataframes(
        config['input']['detr_csv_path'],
        config['postprocessing']['detr'],
        config
    )
    
    # Ensemble YOLO models
    wbf_yolo = weighted_boxes_fusion_df(
        yolo_df,
        iou_thresh=config['ensemble']['iou_threshold'],
        conf_thresh=config['ensemble']['confidence_threshold']
    )
    
    if config['postprocessing']['ensemble_yolo']['enabled']:
        wbf_yolo = postprocessing_pipeline(
            config['postprocessing']['ensemble_yolo']['config'],
            wbf_yolo
        )
    
    # Ensemble all models
    wbf_all = weighted_boxes_fusion_df(
        pd.concat([wbf_yolo, detr_df], ignore_index=True),
        iou_thresh=config['ensemble']['iou_threshold'],
        conf_thresh=config['ensemble']['confidence_threshold']
    )
    
    if config['postprocessing']['ensemble_all']['enabled']:
        wbf_all = postprocessing_pipeline(
            config['postprocessing']['ensemble_all']['config'],
            wbf_all
        )
    
    # Save results
    output_path = config['output']['path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wbf_all.to_csv(output_path, index=False)
    
    return wbf_all

if __name__ == "__main__":
    ensemble_pipeline("parameters/postprocessing_config_files/ensemble.yaml")