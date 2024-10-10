import cv2
import numpy as np

from skimage import color
from skimage import io


# Define the Gaussian kernel function
def gaussian_kernel(size, sigma):
	size = int(size) // 2
	x, y = np.mgrid[-size:size+1, -size:size+1]
	normal = 1 / (2.0 * np.pi * sigma**2)
	g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
	return g

# Load the image
img = cv2.imread('imageb.png')


# and convert to grayscale
def convert_to_grayscale(img):
    grayscale_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            grayscale_img[i, j] = np.clip(0.2989 * img[i, j, 0] + 0.5870 * img[i, j, 1] + 0.1140 * img[i, j, 2])
    # cv2.imshow('Grayscale Image', grayscale_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return grayscale_img



# Apply Gaussian filter
size = 5
sigma = 1
gaussian_filter_img = cv2.filter2D(convert_to_grayscale(img), -1, gaussian_kernel(size, sigma))

# Display the original and filtered images
cv2.imshow('Original Image', img)
cv2.imshow('Grayscale Image', convert_to_grayscale(img))
#convert_to_grayscale(img)
cv2.imshow('Gaussian Filtered Image', gaussian_filter_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

