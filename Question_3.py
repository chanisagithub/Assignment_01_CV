import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('highlights_and_shadows.jpg')
if img is None:
    raise FileNotFoundError("Could not load highlights_and_shadows.jpg – check the filename/path.")

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)      # l = L* channel (0-255 in OpenCV)

gamma = 0.4
l_gamma = np.array(255 * (l / 255.0) ** gamma, dtype=np.uint8)

# Merge back and convert to BGR
lab_corrected = cv2.merge([l_gamma, a, b])
corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

# Convert both images to grayscale for histogram
original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corrected_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(original_gray.ravel(), bins=256, range=(0, 256), color='gray')
plt.title('Histogram - Original Image')
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(corrected_gray.ravel(), bins=256, range=(0, 256), color='gray')
plt.title('Histogram - Gamma-Corrected Image (γ = 0.4)')
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.grid(True)

plt.tight_layout()
plt.savefig('q3_histograms.png')
plt.show()

cv2.imwrite('q3_original.png', img)
cv2.imwrite('q3_corrected_gamma_0.5.png', corrected)

cv2.imshow('Original (Fig. 3)', img)
cv2.imshow('Gamma Corrected (γ = 0.5)', corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()

