import cv2
import numpy as np

from skimage import color
from skimage import io


# Load the image
img = cv2.imread('imageb.png')

# Define the Gaussian kernel function
def gaussian_kernel(size, sigma):
	size = int(size) // 2
	x, y = np.mgrid[-size:size+1, -size:size+1]
	normal = 1 / (2.0 * np.pi * sigma**2)
	g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
	return g




# and convert to grayscale
def convert_to_grayscale(image):
    grayscale_img = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            grayscale_img[i, j] = np.clip(0.2989 * image[i, j, 0] + 0.5870 * image[i, j, 1] + 0.1140 * image[i, j, 2])
    return grayscale_img



# Apply Gaussian filter
size = 5
sigma = 1
gaussian_filter_img = cv2.filter2D(convert_to_grayscale(img), -1, gaussian_kernel(size, sigma))

def initialize_xfilter(size):

    x_filter = np.zeros(size)
    h, w = size
    x_filter[h//2][0]=-1
    x_filter[h//2][-1]=1

    return x_filter

def initialize_yfilter(size):

    y_filter = np.zeros(size)
    h, w = size
    y_filter[0][h//2]=-1
    y_filter[-1][h//2]=1

    return y_filter


def padding(image):

    padded_image = np.pad(image , ((1,1),(1,1)) , 'constant', constant_values=(0,0) )

    return padded_image


def conv2d(image, ftr):
    s = ftr.shape + tuple(np.subtract(image.shape, ftr.shape) + 1)
    sub_image = np.lib.stride_tricks.as_strided(image, shape = s, strides = image.strides * 2)
    return np.einsum('ij,ijkl->kl', ftr, sub_image)


def main(image_filename):
    # Load the image
    img = cv2.imread(image_filename)

    grayscale_img = convert_to_grayscale(img)
    x_filter = initialize_xfilter((3,3))
    y_filter = initialize_yfilter((3,3))
    # convolve the image with the x filter
    I_x = conv2d(padding(grayscale_img), x_filter)

    # convolve the image with the y filter
    I_y = conv2d(padding(grayscale_img), y_filter)
    # calculate the gradient magnitude
    G = np.sqrt(np.power(I_x,2) + np.power(I_y,2))
    # apply a threshold. It is different for different images.
    G = np.where(G > 66, G, 0)

    cv2.imshow('1.Original Image', img)
    cv2.imshow('2.Grayscale Image', grayscale_img)
    cv2.imwrite('grayscale.png', grayscale_img)
    cv2.imshow('3.Gaussian Filtered Image', gaussian_filter_img)
    cv2.imwrite('gaussian_filter_img.png', gaussian_filter_img)
    cv2.imshow('4.Gradient Magnitude Image', G)
    cv2.imwrite('gradient_magnitude.png', G)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the main function with the image filename
main('imageb.png')

