import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from matplotlib.patches import Patch

# Fungsi untuk load gambar dari folder
def load_images_from_folder(uploaded_files, image_size=(256, 256)):
    images = []
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is not None:
            img = cv2.resize(img, image_size)  # Resize agar lebih cepat
            images.append(img)
    return np.array(images)

# Kelas K-Means dari awal
class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=100):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {j: [] for j in range(self.k)}
            distances = np.linalg.norm(data[:, np.newaxis] - np.array(list(self.centroids.values())), axis=2)
            labels = np.argmin(distances, axis=1)

            for idx, label in enumerate(labels):
                self.classifications[label].append(data[idx])

            prev_centroids = dict(self.centroids)
            for classification in self.classifications:
                self.centroids[classification] = np.mean(self.classifications[classification], axis=0)

            optimized = True
            for c in self.centroids:
                if np.sum((self.centroids[c] - prev_centroids[c]) / prev_centroids[c] * 100.0) > self.tol:
                    optimized = False
            if optimized:
                break

    def predict(self, data):
        distances = np.linalg.norm(data - np.array(list(self.centroids.values())), axis=1)
        return np.argmin(distances)

# Preprocessing gambar
def preprocess_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vectorized = img_rgb.reshape((-1, 3))
    return np.float32(vectorized)

# Fungsi untuk menghitung silhouette score dan inertia
def compute_eval_metrics(vectorized, label, center):
    silhouette = silhouette_score(vectorized, label)
    inertia = np.sum((vectorized - center[label.flatten()]) ** 2)  # Hitung inertia
    return silhouette, inertia

# Visualisasi hasil clustering
def visualize_results(original_image, segmented_images, K_values, img_index, cluster_centers_list, silhouettes, inertias):
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    figure_size = 20
    plt.figure(figsize=(figure_size, figure_size))

    plt.subplot(1, 4, 1)
    plt.imshow(original_image_rgb)
    plt.title(f'Original Image {img_index + 1}')
    plt.xticks([]) 
    plt.yticks([])

    for i, (segmented_image, K, cluster_centers) in enumerate(zip(segmented_images, K_values, cluster_centers_list)):
        plt.subplot(1, 4, i + 2)
        plt.imshow(segmented_image)
        plt.title(f'Segmented (K={K})\nSilhouette: {silhouettes[i]:.2f}, Inertia: {inertias[i]:.2f}')
        plt.xticks([])
        plt.yticks([])

        # Create legend for cluster colors
        patches = [Patch(facecolor=np.array(center)/255, label=f'Cluster {j+1}') for j, center in enumerate(cluster_centers)]
        plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.2, 1))

    plt.tight_layout()
    st.pyplot(plt)

# Proses gambar
def process_images(images, K_values=[2, 3, 4]):
    for img_index, original_image in enumerate(images):
        vectorized = preprocess_image(original_image)  # Flatten and vectorize the image
        segmented_images = []
        cluster_centers_list = []
        silhouettes = []
        inertias = []

        for K in K_values:
            model = K_Means(k=K)
            model.fit(vectorized)
            label = np.array([model.predict(point) for point in vectorized])
            center = np.array([model.centroids[i].astype(np.uint8) for i in model.centroids])

            segmented_image = np.zeros_like(original_image)

            # Reconstruct the segmented image from the labels and centers
            for i in range(K):
                mask = (label.flatten() == i)
                mask = mask.reshape(original_image.shape[:2])
                color = np.uint8(center[i])
                segmented_image[mask] = color

            segmented_images.append(segmented_image)
            cluster_centers_list.append(center)  # Store cluster centers (colors)

            # Compute evaluation metrics
            silhouette, inertia = compute_eval_metrics(vectorized, label, center)
            silhouettes.append(silhouette)
            inertias.append(inertia)

        visualize_results(original_image, segmented_images, K_values, img_index, cluster_centers_list, silhouettes, inertias)

# Custom CSS for background color and centered text
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #FFD1DC, #F0E68C); /* Pink pastel to light yellow gradient */
    }
    h1 {
        text-align: center;
    }
    p {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Centered Title
st.markdown("<h1>Aerial Image Clustering using K-Means</h1>", unsafe_allow_html=True)

# Centered Subheading
st.markdown("""
<p>
Created by Group 3 consisting of Aliya Sania (210035), Sarah Khairunnisa Prihantoro (210063), and Zakia Noorardini (210065) 
to fulfill the Midterm Exam for Data Mining 2024.
</p>
""", unsafe_allow_html=True)


# Upload images
uploaded_files = st.file_uploader("Upload Image Files", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
k_values = st.multiselect('Select K values for K-Means', options=[2, 3, 4, 5, 6], default=[2, 3])

if uploaded_files and k_values:
    images = load_images_from_folder(uploaded_files)
    st.write(f"Total images uploaded: {len(images)}")
    process_images(images, K_values=k_values)
