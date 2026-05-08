import cv2
import matplotlib.pyplot as plt

# Helper function to display images
def show_image(img, title, is_gray=False):
    if not is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
        return
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load the image
image = cv2.imread('Parrot.jpg')  # Change path accordingly

if image is None:
    print("Error: Could not load image.")
else:
    # Show original
    show_image(image, "Original Image (RGB)")

    # 1. RGB to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    show_image(hsv_image, "HSV Image")

    # 2. RGB to HSL (in OpenCV, this is called HLS)
    hsl_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    show_image(hsl_image, "HSL Image")

    # 3. RGB to YCrCb
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    show_image(ycrcb_image, "YCrCb Image")

    # 4. RGB to Lab
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    show_image(lab_image, "Lab Image")

    # 5. RGB to XYZ
    xyz_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    show_image(xyz_image, "XYZ Image")

    # 6. Convert to Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image(gray_image, "Grayscale Image", is_gray=True)

    # First: Grayscale -> BGR
    gray_to_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    gray_to_hsv = cv2.cvtColor(gray_to_bgr, cv2.COLOR_BGR2HSV)
    gray_to_lab = cv2.cvtColor(gray_to_bgr, cv2.COLOR_BGR2Lab)

    # Display results
    show_image(gray_to_bgr, "Grayscale + BGR")
    show_image(gray_to_hsv, "Grayscale + HSV (via BGR)")
    show_image(gray_to_lab, "Grayscale + Lab (via BGR)")