import cv2
import numpy as np
import matplotlib.pyplot as plt

# Helper function to show image and histogram
def show_image_and_histogram(img, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(title)
    axes[0].axis('off')

    axes[1].hist(img.ravel(), bins=256, range=(0, 256), color='black')
    axes[1].set_title(f'{title} Histogram')
    axes[1].set_xlim([0, 256])

    plt.tight_layout()
    plt.show()

# Load image and convert to grayscale
image = cv2.imread('Einstein.jpg')
if image is None:
    print("Error: Image not loaded. Check file name/path.")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Original grayscale
    show_image_and_histogram(gray, "Original Grayscale Image")

    # 2. Histogram Equalization
    hist_eq = cv2.equalizeHist(gray)
    show_image_and_histogram(hist_eq, "Histogram Equalization")

    # 3. Contrast Stretching
    min_val = np.min(gray)
    max_val = np.max(gray)
    contrast_stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    show_image_and_histogram(contrast_stretched, "Contrast Stretching")

    # 4. Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adaptive_eq = clahe.apply(gray)
    show_image_and_histogram(adaptive_eq, "Adaptive Histogram Equalization (CLAHE)")