from math import log10, sqrt
import cv2
import numpy as np

def PSNR(ground_path, output_path):
    ground = cv2.imread(ground_path)
    output= cv2.imread(output_path)
    mse = np.mean((ground - output) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
