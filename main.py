import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the image
image_path = 'input_image.jpg'  # Change this to your actual image filename
image = cv2.imread(image_path)
if image is None:
    print("Failed to load image. Check the path.")
    exit()

# Convert image from BGR (OpenCV) to RGB (for visualization)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
pixels = image_rgb.reshape((-1, 3))

# Ask user for number of clusters
k = int(input("Enter number of color clusters (e.g., 3 or 5): "))

# Apply KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels)

# Get the cluster centers (colors) and labels
colors = np.array(kmeans.cluster_centers_, dtype='uint8')
labels = kmeans.labels_

# Create segmented image (replace each pixel by its cluster center)
segmented_img = colors[labels].reshape(image_rgb.shape)

# Create a bar showing the dominant colors
def create_color_bar(colors):
    bar_height = 50
    bar = np.zeros((bar_height, len(colors) * 50, 3), dtype='uint8')
    for idx, color in enumerate(colors):
        start_x = idx * 50
        end_x = start_x + 50
        bar[:, start_x:end_x, :] = color
    return bar

color_bar = create_color_bar(colors)

# Display everything
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Segmented Image")
plt.imshow(segmented_img)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Dominant Colors")
plt.imshow(color_bar)
plt.axis('off')

plt.tight_layout()
plt.show()
