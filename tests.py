import numpy as np
from skimage.metrics import structural_similarity

def PSNR (original, processed):
    error = np.mean((original - processed)**2)
    if error == 0:
        return 100
    PIXEL_MAX = 255.0
    return (20*np.log10(PIXEL_MAX/np.sqrt(error)))

def SSIM (original, processed):
    return (structural_similarity (original, processed, multichannel=True))
    