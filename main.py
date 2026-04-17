import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('emma.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Could not load the image. Check the path.")

# Breakpoints
x_points = np.array([0, 50, 50, 150, 150, 255])
y_points = np.array([0, 50,100, 255,150, 255])

# Create LUT using linear interpolation (exactly reproduces the plot)
lut = np.interp(np.arange(256), x_points, y_points).astype(np.uint8)

transformed = cv2.LUT(img, lut)