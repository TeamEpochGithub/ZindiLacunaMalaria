from typing import List

import cv2


def yolo_predict(model, image: cv2.Mat, conf: float = 0.0) -> List[dict]:
    """Helper function to get YOLO predictions"""
    preds = model.predict(image, augment=False, conf=conf, verbose=False)
    predictions = []
    for pred in preds:
        if len(pred) > 0:
            for prediction in pred:
                xmin, xmax, ymin, ymax = prediction.boxes.xyxy[0].tolist()
                confidence = prediction.boxes.conf.item()
                class_id = int(prediction.boxes.cls.item())
                predictions.append({
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                    'class': class_id,
                    'confidence': confidence
                })
    return predictions
