import cv2
import numpy as np
import matplotlib.pyplot as plt

brain = cv2.imread('brain_pd.png', cv2.IMREAD_GRAYSCALE)
if brain is None:
    raise FileNotFoundError("Could not load brain_pd.png. Check the filename/path.")


# Breakpoints for white-matter accentuation (emphasize high-intensity range)
x_white = np.array([0,   110,  175,  255])
y_white = np.array([0,    20,  255,  255])

# Breakpoints for gray-matter accentuation (emphasize mid-intensity range)
x_gray = np.array([0,   80,  120,  160,  255])
y_gray = np.array([0,   40,  130,  240,  255])

lut_white = np.interp(np.arange(256), x_white, y_white).astype(np.uint8)
lut_gray = np.interp(np.arange(256), x_gray, y_gray).astype(np.uint8)

# Apply
brain_white = cv2.LUT(brain, lut_white)
brain_gray = cv2.LUT(brain, lut_gray)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(brain.ravel(), bins=256, range=(0, 256), color='gray')
axes[0].set_title('Histogram of Brain PD Image')
axes[0].set_xlabel('Intensity')
axes[0].set_ylabel('Pixel Count')
axes[0].grid(True)

axes[1].imshow(brain_white, cmap='gray')
axes[1].set_title('White-Matter Accentuated Image')
axes[1].axis('off')

axes[2].imshow(brain_gray, cmap='gray')
axes[2].set_title('Gray-Matter Accentuated Image')
axes[2].axis('off')

fig.tight_layout()
fig.savefig('q2_histogram.png')

intensity_values = np.arange(256)
fig_tf, ax_tf = plt.subplots(figsize=(8, 5))
ax_tf.plot(intensity_values, intensity_values, 'k--', label='Identity')
ax_tf.plot(intensity_values, lut_white, color='tab:blue', label='White-Matter Transform')
ax_tf.plot(intensity_values, lut_gray, color='tab:orange', label='Gray-Matter Transform')
ax_tf.set_title('Intensity Transformations')
ax_tf.set_xlabel('Input Intensity')
ax_tf.set_ylabel('Output Intensity')
ax_tf.set_xlim(0, 255)
ax_tf.set_ylim(0, 255)
ax_tf.grid(True)
ax_tf.legend()
fig_tf.tight_layout()
fig_tf.savefig('q2_transformation_plot.png')

plt.show()
