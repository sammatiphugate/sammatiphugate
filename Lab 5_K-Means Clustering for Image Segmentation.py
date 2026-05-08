import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Man.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixel_vals = image_rgb.reshape((-1, 3))
pixel_vals = np.float32(pixel_vals)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
K = 4
_, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(image_rgb.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(segmented_image)
axes[1].set_title(f"K-Means Segmented (K={K})")
axes[1].axis('off')

plt.tight_layout()
plt.show()