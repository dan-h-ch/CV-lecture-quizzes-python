import cv2

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


def surf(data):
    y = np.arange(0, data.shape[0])
    x = np.arange(0, data.shape[1])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(X, Y, data, rstride=1, cstride=1, linewidth=0,
                    cmap='jet', antialiased=False)

    plt.show(block=False)


def LoG(size, sigma):
    x = y = np.linspace(-size, size, 2*size+1)
    x, y = np.meshgrid(x, y)

    f = (x**2 + y**2)/(2*sigma**2)
    k = -1./(np.pi * sigma**4) * (1 - f) * np.exp(-f)

    return k


# Edge demo

# Read Lena image
lenaL = cv2.imread('images/lena.png')
height, width = lenaL.shape[:2]
cv2.imshow('Original image, color', lenaL)

# Convert to monochrome (grayscale) using BGR2GRAY.
lenaMono = cv2.cvtColor(lenaL, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original image, monochrome', lenaMono)

# Make a blurred/smoothed version. Use cv2.getGaussianKernel to get the h kernel
# Create a Gaussian filter
filter_size = 11
filter_sigma = 4
h = cv2.getGaussianKernel(filter_size, filter_sigma)
h = h * h.T
print( h)

# Mimic Matlab's surf(h)
surf(h)

# Use cv2.filter2D with BORDER_CONSTANT to get results similar to the Matlab demo
lenaSmooth = cv2.filter2D(lenaMono, -1, h, borderType=cv2.BORDER_CONSTANT)
cv2.imshow('Smoothed image', lenaSmooth)

# Method 1: Shift left and right, and show diff image
lenaL = np.copy(lenaSmooth)  # Let's use np.copy to avoid modifying the original array
lenaL[:, :-1] = lenaL[:, 1:]

lenaR = np.copy(lenaSmooth)  # Let's use np.copy to avoid modifying the original array
lenaR[:, 1:] = lenaR[:, :-1]

lenaDiff = 1. * lenaR - 1. * lenaL  # Multiplying by 1. as a shortcut to converting array to float

# Here we shift the value range to fit [0, 255] and make sure the data type is uint8 in order to display the results.
# normalizedImg = np.zeros((height, width))
lenaDiff = cv2.normalize(lenaDiff, dst=lenaDiff, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
cv2.imshow('Difference between right and left shifted images', lenaDiff.astype(np.uint8))

# Method 2: Canny edge detector
# OpenCV doesn't have a function similar to edge but it does have a Canny Edge detector
# OpenCV needs you to specify low and high threshold values. While these are not the
# exactly the same as the ones used in the demo you should refer to the lines below
# as a reference on how cv2.Canny works
thresh1 = 60
thresh2 = 50

cannyEdges = cv2.Canny(lenaMono, thresh2, thresh1)
cv2.imshow('Original edges thresh2={thresh2} thresh1={thresh1}'.format(thresh1=thresh1, thresh2=thresh2), cannyEdges)

cannyEdges = cv2.Canny(lenaSmooth, thresh2, thresh1)
cv2.imshow('Edges of smoothed image thresh2={thresh2} thresh1={thresh1}'.format(thresh1=thresh1, thresh2=thresh2), cannyEdges)

# Method 3: Laplacian of Gaussian
h = LoG(4, 1.)
surf(h)

# Let's use cv2.filter2D with the new h
logEdges = cv2.filter2D(1. * lenaMono, -1, h, borderType=cv2.BORDER_CONSTANT) 
logEdgesShow = cv2.normalize(logEdges, dst=logEdges, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

cv2.imshow('Laplacian image before zero crossing', logEdgesShow.astype(np.uint8))

# OpenCV doesn't have a function edge like Matlab that implements a 'log' method. This would
# have to be implemented from scratch. This may take a little more time to implement this :).

cv2.waitKey(0)
