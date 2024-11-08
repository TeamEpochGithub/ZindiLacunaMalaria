import cv2

from inference.tta import get_img_tta_augs

im = cv2.imread('test.jpeg')

for im2 in get_img_tta_augs(im):
    cv2.imshow('test', im2)
    cv2.waitKey(0)

print(len(get_img_tta_augs(im)))