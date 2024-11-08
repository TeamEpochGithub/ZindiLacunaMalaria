from typing import List

import cv2
import pandas as pd
import tqdm

from inference.tta import get_img_tta_augs


def yolo_predict_tta(model, img_paths, conf: float = 0.0) -> List[pd.DataFrame]:
    tta_preds = [[] for _ in range(4)]

    for image_dir in tqdm.tqdm(img_paths, desc='predicting with YOLO'):
        image_id = image_dir.split('/')[-1]
        image = cv2.imread(image_dir)

        for i, im_aug in enumerate(get_img_tta_augs(image)):
            predictions = model(im_aug, conf=conf, device=0, verbose=False, augment=False)

            final_predictions = []
            for r in predictions:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()

                    class_name = model.names[cls]

                    final_predictions.append({
                        'Image_ID': image_id,
                        'class': class_name,
                        'confidence': conf,
                        'ymin': xyxy[1],
                        'xmin': xyxy[0],
                        'ymax': xyxy[3],
                        'xmax': xyxy[2]
                    })

                if len(boxes) == 0:
                    final_predictions.append({
                        'Image_ID': image_id,
                        'class': 'NEG',
                        'confidence': 0,
                        'ymin': 0,
                        'xmin': 0,
                        'ymax': 0,
                        'xmax': 0
                    })

            tta_preds[i] += [f.values() for f in final_predictions]
            break

    tta_preds = [pd.DataFrame(aug_preds, columns=['Image_ID', 'class', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
                 for aug_preds in tta_preds]

    return tta_preds