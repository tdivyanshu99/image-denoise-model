import os
import time
import bm3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from noise import add_noise
from tests import PSNR, SSIM

def grayscale(img):
	return np.dot(img[...,:3], [0.299, 0.587, 0.114])

def median_filter(data, kernel_size):
    temp = []
    indexer = kernel_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):
        for j in range(len(data[0])):
            for k in range(kernel_size):
                if i + k - indexer < 0 or i + k - indexer > len(data) - 1:
                    for c in range(kernel_size):
                        temp.append(0)
                else:
                    if j + k - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for x in range(kernel_size):
                            temp.append(data[i + k - indexer][j +  x - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final

images = ["dataset/pixel.jpg", "dataset/1to1.jpg", "dataset/fractal.jpg", "dataset/lena.jpg", "dataset/nature.jpg"]
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
    # denoise = bm3d.bm3d(noisy_img, sigma_psd=.1, stage_arg=noisy_img)
    denoised_img = median_filter(noisy_img, 3)
    end = time.time()
    times_list.append(end - start)

    psnr_list.append(PSNR(img, denoised_img))
    ssim_list.append(SSIM(img, denoised_img))
    psnr_noisy_list.append(PSNR(img, noisy_img))
    ssim_noisy_list.append(SSIM(img, noisy_img))


    display = [img, noisy_img, denoised_img]
    label = ['Original Image', 'Noise', 'Filter applied']
    fig = plt.figure()

    for i in range(len(display)):
            fig.add_subplot(1, 3, i+1)
            plt.imshow(display[i], cmap = 'gray')
            plt.title(label[i])

    # print(img.shape)
    # print(noisy_img.shape)
    # print(denoised_img.shape)


    # plt.show()
    # image=image.replace("dataset/", "processed/")
    # matplotlib.image.imsave(image, denoised_img, cmap='gray')


print('PSNR: ', psnr_list)
print('SSIM: ', ssim_list)
print('Times: ', times_list)
print('PSNR noisy: ', psnr_noisy_list)
print('SSIM noisy: ', ssim_noisy_list)