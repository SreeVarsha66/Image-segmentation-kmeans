# Image Segmentation using K-Means Clustering

This project performs image segmentation using the K-Means clustering algorithm to group similar color pixels in an image. It allows users to choose the number of clusters and then visualizes the segmented image along with the original and a bar of dominant colors.

## 📌 Features
- Loads an input image and reshapes it for clustering
- Uses Scikit-learn's KMeans for color-based clustering
- Allows dynamic selection of number of clusters (`k`)
- Displays:
  - Original Image
  - Segmented Image
  - Dominant Colors bar

## 🛠️ Technologies Used
- Python
- OpenCV (`cv2`)
- NumPy
- scikit-learn
- Matplotlib

## 🚀 How to Run

1. Place your image in the project directory and update the path in `image_path`.
2. Run the Python script.
3. Enter the number of clusters when prompted (e.g., 3, 5).
4. View the segmented image and color distribution.

## 📁 Project Structure
- main.py
- requirements.txt
- input_image.jpg
- output_image.jpg
