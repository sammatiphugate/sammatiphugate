import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from scipy.stats import skew

image = cv2.imread('Building.jpg')  # Update this path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ── 1. GLCM Features ──────────────────────────────────────────────────────────
def extract_glcm_features(gray_img):
    glcm = graycomatrix(gray_img, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    contrast    = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy      = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return contrast, correlation, energy, homogeneity

contrast, correlation, energy, homogeneity = extract_glcm_features(gray)
print("GLCM Features:")
print(f"Contrast: {contrast}, Correlation: {correlation}, "
      f"Energy: {energy}, Homogeneity: {homogeneity}")

# ── 2. Color Histogram ────────────────────────────────────────────────────────
def extract_color_histogram(image_rgb):
    hist_features = []
    for i, col in enumerate(['R', 'G', 'B']):
        hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    return hist_features

color_hist_features = extract_color_histogram(image_rgb)
print(f"Color Histogram Feature Vector Length: {len(color_hist_features)}")

# ── 3. Local Binary Pattern (LBP) ─────────────────────────────────────────────
radius   = 1
n_points = 8 * radius
lbp      = local_binary_pattern(gray, n_points, radius, method='uniform')

lbp_hist, _ = np.histogram(lbp.ravel(),
                            bins=np.arange(0, n_points + 3),
                            range=(0, n_points + 2))
lbp_hist = lbp_hist.astype("float")
lbp_hist /= (lbp_hist.sum() + 1e-7)
print(f"LBP Feature Vector Length: {len(lbp_hist)}")

# ── 4. HOG Features ───────────────────────────────────────────────────────────
hog_features, hog_image = hog(
    gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True,
    transform_sqrt=True
)
print(f"HOG Feature Vector Length: {len(hog_features)}")

# ── Visualizations ────────────────────────────────────────────────────────────
plt.figure(figsize=(6, 4))
plt.imshow(lbp, cmap='gray')
plt.title("Local Binary Pattern")
plt.axis('off')
plt.show()

plt.figure(figsize=(6, 4))
plt.imshow(hog_image, cmap='gray')
plt.title("HOG Features")
plt.axis('off')
plt.show()