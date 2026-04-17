import cv2
import numpy as np
import matplotlib.pyplot as plt

SHOW_PREVIEW = False

img = cv2.imread('emma.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Could not load the image. Check the path.")

# Breakpoints
x_points = np.array([0, 50, 50, 150, 150, 255])
y_points = np.array([0, 50,100, 255,150, 255])

# Create LUT using linear interpolation (exactly reproduces the plot)
lut = np.interp(np.arange(256), x_points, y_points).astype(np.uint8)

transformed = cv2.LUT(img, lut)

# Save for the report
cv2.imwrite('q1_original.png', img)
cv2.imwrite('q1_transformed.png', transformed)

plt.figure(figsize=(8, 6))
plt.plot(range(256), lut, 'b-', linewidth=2)
plt.title('Intensity Transformation (matches Fig. 1a)')
plt.xlabel('Input intensity')
plt.ylabel('Output intensity')
plt.grid(True)
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.savefig('q1_transformation_plot.png')

if SHOW_PREVIEW:
    cv2.imshow('Original (Fig 1b)', img)
    cv2.imshow('After Intensity Transformation', transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.show()
