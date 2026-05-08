import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Leaf.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def show_edge(title, edge_image):
    plt.figure(figsize=(5, 5))
    plt.imshow(edge_image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Canny Edge Detection
canny_edges = cv2.Canny(gray, 100, 200)
show_edge("Canny Edge Detection", canny_edges)

# Sobel Edge Detection
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # X-direction
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Y-direction
sobel_combined = cv2.magnitude(sobelx, sobely)

show_edge("Sobel X", np.abs(sobelx))
show_edge("Sobel Y", np.abs(sobely))
show_edge("Sobel Combined Magnitude", sobel_combined)

# Laplacian Edge Detection
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
show_edge("Laplacian Edge Detection", np.abs(laplacian))