import pandas as pd
import cv2
import os
from tqdm import tqdm


def inference_yolo_model(model, image_paths: list[str], tta: bool = False, conf=0.0) -> dict:
    if not tta:
        # No TTA: standard prediction
        all_preds = []
        for image_path in image_paths:
            image_id = os.path.basename(image_path)

            preds = model.predict(image_path, augment=False, conf=conf, verbose=False)

            for pred in tqdm(preds):
                if len(pred) > 0:
                    for prediction in pred:
                        xmin, xmax, ymin, ymax = prediction.boxes.xyxy[0].tolist()
                        confidence = prediction.boxes.conf.item()
                        class_id = int(prediction.boxes.cls.item())
                        all_preds.append({
                            'Image_ID': image_id,
                            'xmin': xmin,
                            'xmax': xmax,
                            'ymin': ymin,
                            'ymax': ymax,
                            'class': class_id,
                            'confidence': confidence
                        })

        return pd.DataFrame(all_preds)

    else:
        # TTA: Prepare dictionaries to store predictions for each augmentation type
        preds_original = []
        preds_horizontal = []
        preds_vertical = []
        preds_both = []

        # Process each image
        for image_path in tqdm(image_paths):
            image_id = os.path.basename(image_path)
            image = cv2.imread(image_path)

            # Original image
            orig_preds = model.predict(image_path, augment=False, conf=conf, verbose=False)
            for pred in orig_preds:
                if len(pred) > 0:
                    for prediction in pred:
                        xmin, xmax, ymin, ymax = prediction.boxes.xyxy[0].tolist()
                        confidence = prediction.boxes.conf.item()
                        class_id = int(prediction.boxes.cls.item())
                        preds_original.append({
                            'Image_ID': image_id,
                            'xmin': xmin,
                            'xmax': xmax,
                            'ymin': ymin,
                            'ymax': ymax,
                            'class': class_id,
                            'confidence': confidence
                        })

            # Horizontal flip
            h_flipped = cv2.flip(image, 1)  # 1 means horizontal flip
            h_flip_preds = model.predict(h_flipped, augment=False, conf=conf, verbose=False)
            for pred in h_flip_preds:
                if len(pred) > 0:
                    for prediction in pred:
                        xmin, xmax, ymin, ymax = prediction.boxes.xyxy[0].tolist()
                        confidence = prediction.boxes.conf.item()
                        class_id = int(prediction.boxes.cls.item())
                        # Adjust bounding boxes for horizontal flip
                        xmin, xmax = image.shape[1] - xmax, image.shape[1] - xmin
                        preds_horizontal.append({
                            'Image_ID': image_id,
                            'xmin': xmin,
                            'xmax': xmax,
                            'ymin': ymin,
                            'ymax': ymax,
                            'class': class_id,
                            'confidence': confidence
                        })

            # Vertical flip
            v_flipped = cv2.flip(image, 0)  # 0 means vertical flip
            v_flip_preds = model.predict(v_flipped, augment=False, conf=conf, verbose=False)
            for pred in v_flip_preds:
                if len(pred) > 0:
                    for prediction in pred:
                        xmin, xmax, ymin, ymax = prediction.boxes.xyxy[0].tolist()
                        confidence = prediction.boxes.conf.item()
                        class_id = int(prediction.boxes.cls.item())
                        # Adjust bounding boxes for vertical flip
                        ymin, ymax = image.shape[0] - ymax, image.shape[0] - ymin
                        preds_vertical.append({
                            'Image_ID': image_id,
                            'xmin': xmin,
                            'xmax': xmax,
                            'ymin': ymin,
                            'ymax': ymax,
                            'class': class_id,
                            'confidence': confidence
                        })

            # Both horizontal and vertical flip
            both_flipped = cv2.flip(image, -1)  # -1 means both horizontal and vertical flip
            both_flip_preds = model.predict(both_flipped, augment=False, conf=conf, verbose=False)
            for pred in both_flip_preds:
                if len(pred) > 0:
                    for prediction in pred:
                        xmin, xmax, ymin, ymax = prediction.boxes.xyxy[0].tolist()
                        confidence = prediction.boxes.conf.item()
                        class_id = int(prediction.boxes.cls.item())
                        # Adjust bounding boxes for both flips
                        xmin, xmax = image.shape[1] - xmax, image.shape[1] - xmin
                        ymin, ymax = image.shape[0] - ymax, image.shape[0] - ymin
                        preds_both.append({
                            'Image_ID': image_id,
                            'xmin': xmin,
                            'xmax': xmax,
                            'ymin': ymin,
                            'ymax': ymax,
                            'class': class_id,
                            'confidence': confidence
                        })

        # Return a dictionary of DataFrames
        return {
            'original': pd.DataFrame(preds_original),
            'horizontal_flip': pd.DataFrame(preds_horizontal),
            'vertical_flip': pd.DataFrame(preds_vertical),
            'both_flips': pd.DataFrame(preds_both)
        }


if __name__ == '__main__':
    from ultralytics import YOLO
    os.chdir('../..')
    df = pd.read_csv('data/csv_files/Test.csv')
    model = YOLO('model.pt')
    paths = [f'data/img/{img_id}' for img_id in df['Image_ID'].unique()]
    print(inference_yolo_model(model, image_paths=paths, tta=True))
