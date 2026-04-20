import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('einstein.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Could not load einstein.png – check the filename/path.")

# Sobel_x kernel
kernel_sobel_x = np.array([[ 1,  0, -1],
                           [ 2,  0, -2],
                           [ 1,  0, -1]], dtype=np.float32)

sobel_a = cv2.filter2D(img, cv2.CV_32F, kernel_sobel_x)

# Convert to displayable image (absolute value + scale)
sobel_a = cv2.convertScaleAbs(sobel_a)
cv2.imwrite('q6_a_filter2D.png', sobel_a)

def convolve2d_manual(image, kernel):
    # Pad the image
    pad = kernel.shape[0] // 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    result = np.zeros_like(image, dtype=np.float32)
    kh, kw = kernel.shape
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return result

# Same kernel as above
kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)

sobel_b = convolve2d_manual(img, kernel)
sobel_b = cv2.convertScaleAbs(sobel_b)
cv2.imwrite('q6_b_own_code.png', sobel_b)

# Separable kernels
vertical_kernel   = np.array([[1], [2], [1]], dtype=np.float32)   # 3x1
horizontal_kernel = np.array([[1, 0, -1]], dtype=np.float32)      # 1x3

# Two 1D convolutions (order does not matter for this kernel)
temp = cv2.filter2D(img, cv2.CV_32F, vertical_kernel)
sobel_c = cv2.filter2D(temp, cv2.CV_32F, horizontal_kernel)

sobel_c = cv2.convertScaleAbs(sobel_c)
cv2.imwrite('q6_c_separable.png', sobel_c)

# Side-by-side comparison
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1); plt.imshow(img, cmap='gray'); plt.title('Original (Fig. 6)'); plt.axis('off')
plt.subplot(1, 4, 2); plt.imshow(sobel_a, cmap='gray'); plt.title('(a) filter2D'); plt.axis('off')
plt.subplot(1, 4, 3); plt.imshow(sobel_b, cmap='gray'); plt.title('(b) Own code'); plt.axis('off')
plt.subplot(1, 4, 4); plt.imshow(sobel_c, cmap='gray'); plt.title('(c) Separable'); plt.axis('off')
plt.tight_layout()
plt.savefig('q6_all_results.png')
plt.show()