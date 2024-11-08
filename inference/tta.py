from typing import List, Dict

import cv2


def get_img_tta_augs(image: cv2.Mat) -> List[cv2.Mat]:
    tta_images = [image, cv2.flip(image, 0), cv2.flip(image, 1),
                  cv2.flip(cv2.flip(image, 0), 1)]

    return tta_images