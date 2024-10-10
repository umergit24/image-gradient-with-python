import cv2
import numpy as np


def make_gaussian(size, sigma):
    half = int(size) // 2
    gaussian = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = i - half
            y = j - half
            gaussian[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return gaussian

def to_gray(img):
    gray = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gray[i, j] = min(max(0.2989 * img[i, j, 0] + 0.5870 * img[i, j, 1] + 0.1140 * img[i, j, 2], 0), 255)
    return gray

def blur_gaussian(img, size=5, sigma=1):
    gray = to_gray(img)
    kernel = make_gaussian(size, sigma)
    return cv2.filter2D(gray, -1, kernel)

def x_filter(size):
    f = np.zeros(size)
    f[size[0]//2, 0] = -1
    f[size[0]//2, -1] = 1
    return f

def y_filter(size):
    f = np.zeros(size)
    f[0, size[1]//2] = -1
    f[-1, size[1]//2] = 1
    return f

def pad_img(img):
    return np.pad(img, ((1,1),(1,1)), 'constant')

def convolve(img, kernel):
    h, w = img.shape[0] - kernel.shape[0] + 1, img.shape[1] - kernel.shape[1] + 1
    result = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            result[i, j] = np.sum(img[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return result

def process_image(filename):
    img = cv2.imread(filename)
    blurred = blur_gaussian(img)
    gray = to_gray(img)

    Ix = convolve(pad_img(gray), x_filter((3,3)))
    Iy = convolve(pad_img(gray), y_filter((3,3)))

    magnitude = np.sqrt(Ix**2 + Iy**2)
    magnitude[magnitude <= 66] = 0
    phase = np.arctan2(Iy, Ix)
    phase_deg = np.degrees(phase)
    phase_deg[phase_deg <= 10] = 0

    cv2.imshow('Original', img)
    cv2.imshow('Gray', gray)
    cv2.imwrite('images/gray.png', gray)
    cv2.imshow('Blurred', blurred)
    cv2.imwrite('images/blurred.png', blurred)
    # cv2.imshow('Ix', Ix)
    # cv2.imwrite('images/x.png', Ix)
    # cv2.imshow('Iy', Iy)
    # cv2.imwrite('images/y.png', Iy)
    cv2.imshow('Magnitude', magnitude)
    cv2.imwrite('images/magnitude.png', magnitude)
    cv2.imshow('Phase', phase_deg)
    cv2.imwrite('images/phase.png', phase_deg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

process_image('images/imageb.png')