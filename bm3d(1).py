import bm3d
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from noise import add_noise
from tests import PSNR, SSIM

def grayscale(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

images = ["dataset/pixel.jpg", "dataset/1to1.jpg", "dataset/fashion.jpg", "dataset/lena.jpg", "dataset/nature.jpg"]
times_list = []
ssim_list = []
psnr_list = []
psnr_noisy_list = []
ssim_noisy_list = []
 
for image in images:
    image = os.path.join(image) 
    img = grayscale(plt.imread(image))

    noisy_img = add_noise(img, 0.1)

    start = time.time()
    denoised_img = bm3d.bm3d(noisy_img, sigma_psd=1, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    end = time.time()
    times_list.append(end - start)

    psnr_list.append(PSNR(img, denoised_img))
    ssim_list.append(SSIM(img, denoised_img))
    # psnr_noisy_list.append(PSNR(img, noisy_img))
    # ssim_noisy_list.append(SSIM(img, noisy_img))


    display = [img, noisy_img, denoised_img]
    label = ['Original Image', 'Noise', 'Filter applied']
    fig = plt.figure()

    for i in range(len(display)):
            fig.add_subplot(1, 3, i+1)
            plt.imshow(display[i], cmap = 'gray')
            plt.title(label[i])

    # plt.show()
    # image=image.replace("dataset/", "processed/")
    # matplotlib.image.imsave(image, denoised_img, cmap='gray')


print('PSNR: ', psnr_list)
print('SSIM: ', ssim_list)
print('Times: ', times_list)
# print('PSNR noisy: ', psnr_noisy_list)
# print('SSIM noisy: ', ssim_noisy_list)