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
# and convert to grayscale
img = cv2.imread('imageb.png')
grayscale_img = color.rgb2gray(io.imread('imageb.png'))
# img = cv2.imread('imageb.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display

# Apply Gaussian filter
size = 5
sigma = 1
gaussian_filter_img = cv2.filter2D(grayscale_img, -1, gaussian_kernel(size, sigma))

# Display the original and filtered images
cv2.imshow('Original Image', img)
cv2.imshow('Grayscale Image', grayscale_img)
cv2.imshow('Gaussian Filtered Image', gaussian_filter_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

