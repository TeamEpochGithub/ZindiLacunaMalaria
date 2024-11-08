import glob
import os

import pandas as pd
from scipy.stats import describe
from tqdm import tqdm

from postprocessing.ensemble import DualEnsemble
from util.save import save_with_negs

ens_yolo = DualEnsemble(form="wbf", iou_threshold=0.7, conf_threshold=0.1,
                        weights=[1, 1, 1, 1],
                        wbf_reduction='mean')

ens_detr = DualEnsemble(form="wbf", iou_threshold=0.5, conf_threshold=0.1,
                        weights=[1],
                        wbf_reduction='mean')

ens_final = DualEnsemble(form="wbf", iou_threshold=0.5, conf_threshold=0.1,
                         weights=[1, 1],
                         wbf_reduction='mean')

predictions_folder = 'data/predictions'
preds_per_model_folder = [f.path for f in os.scandir(predictions_folder) if f.is_dir()]

preds_per_model = {}

for f in preds_per_model_folder:
    preds_per_model[f.split("/")[-1]] = glob.glob(f"{f}/*.csv")

model_preds = []

# ensemble predictions per model
for k, v in tqdm(preds_per_model.items(), desc="Combining model output"):
    preds_dfs = []
    for pred_path in v:
        preds_dfs.append(pd.read_csv(pred_path))

    if k == 'yol':
        model_preds.append(ens_yolo(preds_dfs))
    elif k == 'det':
        model_preds.append(preds_dfs[0])

# ensamble predictions final
final_preds = ens_final(model_preds)

save_with_negs(final_preds)
