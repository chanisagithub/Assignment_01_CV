import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('jeniffer.jpg')
if img is None:
    raise FileNotFoundError("Could not load jeniffer.jpg – check the path.")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Save/display planes as grayscale
cv2.imwrite('q5_h_plane.png', h)
cv2.imwrite('q5_s_plane.png', s)
cv2.imwrite('q5_v_plane.png', v)

# Optional quick view
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(h, cmap='gray'); plt.title('Hue (H)'); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(s, cmap='gray'); plt.title('Saturation (S)'); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(v, cmap='gray'); plt.title('Value (V)'); plt.axis('off')
plt.tight_layout()
plt.savefig('q5_hsv_planes.png')
plt.show()

plane = s
_, mask = cv2.threshold(plane, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# If background is brighter than foreground, uncomment the next line:
mask = cv2.bitwise_not(mask)

cv2.imwrite('q5_mask.png', mask)

# Correct foreground-only histogram (using mask so background pixels are ignored)
hist_fg = cv2.calcHist([v], [0], mask, [256], [0, 256]).flatten()

# (d) Cumulative sum
cdf = np.cumsum(hist_fg)

# (e) Histogram equalization using the standard slide formula
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255.0 / (cdf_m.max() - cdf_m.min())
lut = np.ma.filled(cdf_m, 0).astype(np.uint8)

# Apply LUT ONLY to foreground pixels in V plane
v_equalized = v.copy()
v_equalized[mask == 255] = lut[v[mask == 255]]

hsv_equalized = cv2.merge([h, s, v_equalized])
result = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

cv2.imwrite('q5_original.png', img)
cv2.imwrite('q5_foreground_equalized.png', result)

cv2.imwrite('q5_original.png', img)
cv2.imwrite('q5_h_plane.png', h)
cv2.imwrite('q5_s_plane.png', s)
cv2.imwrite('q5_v_plane.png', v)
cv2.imwrite('q5_mask.png', mask)
cv2.imwrite('q5_result.png', result)