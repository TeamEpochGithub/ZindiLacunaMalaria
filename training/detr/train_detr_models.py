import yaml
from .core.model import get_model, get_processor
from .core.train import train
from pprint import pprint
import torch

def get_trained_detr_models(config_files, dataset_path) -> list:
    trained_detr_models = []
    for config_file in config_files:
        config_dict = dict(yaml.safe_load(open(config_file)))
        model = get_model(config_dict.pop('model'))
        processor = get_processor(config_dict.pop('processor'))
        # device = config_dict['device']
        # train(config_dict, model, processor, data_dir=dataset_path, device=device)
        
        # Load pretrained weights instead of training
        model.load_state_dict(torch.load('/path/to/detr/weights.pth')) # MAKE SURE THE LOADED MODEL WAS TRAINED WITH THE SAME CONFIG
        trained_detr_models.append(model)
    return trained_detr_models
