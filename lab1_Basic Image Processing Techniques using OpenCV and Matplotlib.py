import cv2
import matplotlib.pyplot as plt

# Helper function to display images
def show_image(img, title):
    if len(img.shape) == 3:  # Color image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load the image
#image = cv2.imread(r'lake-mountain-lake-mirroring-water-preview.jpg')  # Use raw string for path
image = cv2.imread(r'/Users/pranay/Downloads/lake-mountain-lake-mirroring-water-preview.jpg')


if image is None:
    print("Error: Could not load image. Check the file path.")
else:
    # 1. Resize the Image
    resized_image = cv2.resize(image, (300, 200))
    show_image(resized_image, "Resized Image (300x200)")

    # 2. Change the shape of the image (aspect ratio)
    aspect_changed = cv2.resize(image, (400, 100))  # Intentional distortion
    show_image(aspect_changed, "Aspect Ratio Changed (400x100)")

    # 3. Rotate the image by specified angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    angle = 45  # You can change this angle
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    show_image(rotated_image, f"Rotated Image ({angle}°)")

    # 4. Flip the image
    flipped_horizontal = cv2.flip(image, 1)
    show_image(flipped_horizontal, "Horizontally Flipped Image")

    flipped_vertical = cv2.flip(image, 0)
    show_image(flipped_vertical, "Vertically Flipped Image")

    # 5. Crop a Region of Interest (ROI)
    roi = image[100:300, 200:400]  # Make sure the indices are within image size
    show_image(roi, "Cropped Region of Interest (ROI)")

    # 6. Convert to Grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()