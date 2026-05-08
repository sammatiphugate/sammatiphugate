import cv2
import numpy as np
import matplotlib.pyplot as plt

# Helper function to show image and histogram
def show_image_and_histogram(image, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(title)
    axes[0].axis('off')

    axes[1].hist(image.ravel(), bins=256, range=(0, 256), color='black')
    axes[1].set_title(f'{title} Histogram')
    axes[1].set_xlim([0, 256])

    plt.tight_layout()
    plt.show()

# Load and convert image to grayscale
image = cv2.imread('Mount.jpg')  # Update path as needed
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Original
show_image_and_histogram(gray, "Original Grayscale Image")

# 2. Image Negative: s = L - 1 - r
negative = 255 - gray
show_image_and_histogram(negative, "Negative Transformation")

# 3. Logarithmic Transformation: s = c * log(1 + r)
gray_float = np.float32(gray)
log_transformed = cv2.normalize(np.log1p(gray_float), None, 0, 255, cv2.NORM_MINMAX)
log_transformed = np.uint8(log_transformed)
show_image_and_histogram(log_transformed, "Logarithmic Transformation")

# 4. Gamma Transformation: s = c * r^y
def gamma_transform(img, gamma):
    img_normalized = img / 255.0
    gamma_corrected = np.power(img_normalized, gamma)
    gamma_corrected = np.uint8(cv2.normalize(gamma_corrected, None, 0, 255, cv2.NORM_MINMAX))
    return gamma_corrected

gamma_1_5 = gamma_transform(gray, 1.5)
gamma_0_5 = gamma_transform(gray, 0.5)

show_image_and_histogram(gamma_1_5, "Gamma Transformation (γ = 1.5)")
show_image_and_histogram(gamma_0_5, "Gamma Transformation (γ = 0.5)")

# 5. Contrast Stretching (Min-Max Normalization)
min_val = np.min(gray)
max_val = np.max(gray)
contrast_stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
show_image_and_histogram(contrast_stretched, "Contrast Stretching")