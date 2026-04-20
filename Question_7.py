import cv2
import numpy as np

def zoom_image(image, scale, method='nearest'):
    """
    Zoom an image by factor s using nearest-neighbor or bilinear interpolation.
    """
    if method == 'nearest':
        interp = cv2.INTER_NEAREST
    elif method == 'bilinear':
        interp = cv2.INTER_LINEAR
    else:
        raise ValueError("method must be 'nearest' or 'bilinear'")

    # New size = (width * s, height * s)
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    zoomed = cv2.resize(image, (new_width, new_height), interpolation=interp)
    return zoomed


def normalized_ssd(img1, img2):
    """Compute normalized SSD = (1 / num_pixels) * sum((img1 - img2)^2)"""
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    ssd = np.sum(diff ** 2)
    num_pixels = img1.shape[0] * img1.shape[1]
    if len(img1.shape) == 3:          # color image
        num_pixels *= img1.shape[2]
    return ssd / num_pixels


# ====================== TEST WITH s = 4 ======================
scale = 4.0

# Load the four images
large1 = cv2.imread('Images/im01.png')
small1 = cv2.imread('Images/im01small.png')
large2 = cv2.imread('Images/im02.png')
small2 = cv2.imread('Images/im02small.png')

if any(im is None for im in [large1, small1, large2, small2]):
    raise FileNotFoundError("One or more test images not found. Check filenames.")

# Zoom the small versions
nn1 = zoom_image(small1, scale, method='nearest')
bl1 = zoom_image(small1, scale, method='bilinear')

nn2 = zoom_image(small2, scale, method='nearest')
bl2 = zoom_image(small2, scale, method='bilinear')

# Compute normalized SSD
ssd_nn1 = normalized_ssd(large1, nn1)
ssd_bl1 = normalized_ssd(large1, bl1)
ssd_nn2 = normalized_ssd(large2, nn2)
ssd_bl2 = normalized_ssd(large2, bl2)

print(f"Image Pair 1 - Nearest-neighbor SSD: {ssd_nn1:.4f}")
print(f"Image Pair 1 - Bilinear SSD:        {ssd_bl1:.4f}")
print(f"Image Pair 2 - Nearest-neighbor SSD: {ssd_nn2:.4f}")
print(f"Image Pair 2 - Bilinear SSD:        {ssd_bl2:.4f}")

# Save zoomed results for the report
cv2.imwrite('q7_small1_nearest_x4.png', nn1)
cv2.imwrite('q7_small1_bilinear_x4.png', bl1)
cv2.imwrite('q7_small2_nearest_x4.png', nn2)
cv2.imwrite('q7_small2_bilinear_x4.png', bl2)