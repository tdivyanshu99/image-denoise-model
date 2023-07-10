import numpy as np
from scipy.signal import convolve2d

# def add_noise(image, sigma=20): #Gaussian noise
#     image = image + np.random.normal(0, sigma, image.shape)
#     return (add_blur(image, 3))

def add_noise(image, sigma=0.1): #Salt-and-Pepper noise
    noisy = image.copy()
    noisy[np.random.rand(*image.shape) < sigma] = 0
    noisy[np.random.rand(*image.shape) < sigma] = 255
    return (add_blur(noisy, 3))

def add_blur(image, kernel_size=3): 
    kernel = np.ones((kernel_size, kernel_size))/(kernel_size**2)
    blurred = convolve2d(image, kernel, mode='same')
    return (blurred)
