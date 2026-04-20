import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('spider.png')
if img is None:
    raise FileNotFoundError("Could not load spider.png – check the path.")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

def vibrance_lut(a=0.65, sigma=70):
    x = np.arange(256, dtype=np.float32)
    boost = a * 128 * np.exp( - (x - 128)**2 / (2 * sigma**2) )
    f_x = np.minimum(x + boost, 255)
    return f_x.astype(np.uint8)

a_value = 0.65
lut_s = vibrance_lut(a=a_value)

s_enhanced = cv2.LUT(s, lut_s)

hsv_enhanced = cv2.merge([h, s_enhanced, v])
enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

x = np.arange(256)
plt.figure(figsize=(8, 6))
plt.plot(x, lut_s, 'b-', linewidth=2)
plt.title(f'Vibrance Transformation (a = {a_value}, σ = 70)')
plt.xlabel('Input Saturation (x)')
plt.ylabel('Output Saturation f(x)')
plt.grid(True)
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.savefig('q4_transformation_plot.png')
plt.show()

cv2.imwrite('q4_original.png', img)
cv2.imwrite(f'q4_vibrance_a_{a_value}.png', enhanced)

cv2.imshow('Original (Fig. 4)', img)
cv2.imshow(f'Vibrance Enhanced (a={a_value})', enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()