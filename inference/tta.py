from typing import List, Dict

import cv2

def model_tta_preprocessing(
        images: List[cv2.Mat],
        tta_transforms: List[str] = None
) -> Dict[str, List[cv2.Mat]]:
    """
    Perform Test Time Augmentation (TTA) on a list of images.

    Args:
        images: List of OpenCV images (cv2.Mat)
        tta_transforms: List of transforms to apply. Options: ['horizontal_flip', 'vertical_flip', 'rotate']

    Returns:
        Dictionary of augmented images for each transform, including the original.
    """
    # Validate and set default TTA transforms if none provided
    valid_transforms = {'horizontal_flip', 'vertical_flip', 'rotate'}
    if tta_transforms is None or not tta_transforms:
        tta_transforms = list(valid_transforms)
    else:
        invalid_transforms = set(tta_transforms) - valid_transforms
        if invalid_transforms:
            raise ValueError(f"Invalid transforms: {invalid_transforms}. "
                             f"Valid options are: {valid_transforms}")

    # Initialize dictionaries for each requested transform
    tta_images = {'original': images.copy()}

    # Apply requested transforms
    for transform in tta_transforms:
        new_tta_images = {}
        for aug_key, aug_images in tta_images.items():
            new_tta_images[aug_key] = aug_images.copy()
            if transform == 'horizontal_flip':
                for i, image in enumerate(aug_images):
                    h_flipped = cv2.flip(image, 1)
                    new_tta_images[f"{aug_key}_horizontal_flip"].append(h_flipped)
            elif transform == 'vertical_flip':
                for i, image in enumerate(aug_images):
                    v_flipped = cv2.flip(image, 0)
                    new_tta_images[f"{aug_key}_vertical_flip"].append(v_flipped)
            elif transform == 'rotate':
                for i, image in enumerate(aug_images):
                    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    new_tta_images[f"{aug_key}_rotate"].append(rotated)
        tta_images.update(new_tta_images)

    return tta_images