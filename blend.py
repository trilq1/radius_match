import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def helperBlendImages(I1, I2):

    # Identify the image regions in the two images by masking out the black regions.
    mask1 = I1.sum(axis=2)
    mask1 = (mask1 > 0).astype(int)

    mask2 = I2.sum(axis=2)
    mask2 = (mask2 > 0).astype(int)

    maskc = ((mask1 + mask2) // 2).astype(bool)

    # Compute alpha values that are proportional to the center seam of the two images.
    alpha1 = (np.ones(mask1.shape, dtype=np.uint8))
    alpha2 = (np.ones(mask2.shape, dtype=np.uint8))
    
    edge1 = edge(mask1)
    
    if np.amin(edge1) == 0 and np.amax(edge1) == 0:
        dist1 = np.ones(edge1.shape)*np.inf

    else:
        dist1 = ndimage.distance_transform_edt(1 - edge1)

    edge2 = edge(mask2)
    dist2 = ndimage.distance_transform_edt(1 - edge2)

    alpha1[maskc] = dist1[maskc] >  dist2[maskc]
    alpha2[maskc] = dist1[maskc] <= dist2[maskc]

    alpha1 = alpha1[..., np.newaxis]
    alpha2 = alpha2[..., np.newaxis]

    outputImage = (alpha1*I1 + alpha2*I2).astype(np.uint8)

    return outputImage

def edge(I):
    kernel = np.ones((7,7), np.uint8)
    map = I - cv2.erode(I.astype(np.uint8), kernel=kernel, iterations=1)
    return map
