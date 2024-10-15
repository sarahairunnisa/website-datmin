import streamlit as st
import numpy as np
from PIL import Image
import cv2

# Warna-warna yang lebih lembut untuk representasi kluster
soft_colors = [
    [173, 216, 230],  # Light Blue
    [119, 158, 203],  # Slate Blue
    [240, 230, 140],  # Khaki
    [176, 224, 230],  # Powder Blue
    [222, 184, 135],  # Burlywood
    [152, 251, 152],  # Pale Green
    [255, 228, 196],  # Bisque
    [216, 191, 216],  # Thistle
    [192, 192, 192],  # Silver
    [255, 182, 193]   # Light Pink
]

# Fungsi untuk menghitung jarak Euclidean antara dua vektor
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Fungsi untuk segmentasi gambar menggunakan K-Means manual
def kmeans_manual(image, k, centroids, max_iters=100):
    # Ubah gambar ke ruang warna LAB agar clustering lebih baik
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Reshape gambar menjadi array dua dimensi
    pixels = lab_image.reshape(-1, 3)

    for _ in range(max_iters):
        # Hitung jarak setiap piksel ke setiap centroid
        distances = np.zeros((pixels.shape[0], k))
        for j in range(k):
            distances[:, j] = np.linalg.norm(pixels - centroids[j], axis=1)

        # Tentukan kluster setiap piksel berdasarkan jarak terdekat ke centroid
        labels = np.argmin(distances, axis=1)

        # Hitung ulang centroid berdasarkan rata-rata piksel di setiap kluster
        new_centroids = np.array([pixels[labels == j].mean(axis=0) for j in range(k)])

        # Jika centroid tidak berubah, keluar dari loop
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    # Buat gambar dengan warna lembut untuk tiap kluster
    segmented_image = np.zeros(image.shape, dtype=np.uint8)
    for i in range(k):
        segmented_image[labels.reshape(image.shape[:2]) == i] = soft_colors[i]

    return segmented_image, centroids

# Setup Streamlit
st.title("Segment Level Clustering dengan Warna Lembut")

# Menu untuk memilih latihan atau pengujian
menu = st.sidebar.selectbox("Pilih Menu", ("Latihan", "Pengujian"))

# Input untuk memilih jumlah cluster (K)
k_value = st.sidebar.number_input("Masukkan jumlah K (cluster)", min_value=2, max_value=10, value=3)

if menu == "Latihan":
    # Input gambar untuk latihan
    uploaded_files = st.sidebar.file_uploader("Upload Gambar untuk Latihan", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Jika file gambar diunggah
    if uploaded_files:
        images = []

        for uploaded_file in uploaded_files:
            # Baca gambar
            img = Image.open(uploaded_file)
            img = img.resize((256, 256))  # Ubah ukuran gambar
            img_np = np.array(img)  # Konversi ke numpy array
            images.append(img_np)

        # Latihan untuk menghitung centroid
        centroids = []
        for img in images:
            lab_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            pixels = lab_image.reshape(-1, 3)
            centroids.append(pixels[np.random.choice(pixels.shape[0], k_value, replace=False)])

        # Rata-rata centroid dari semua gambar
        st.session_state.centroids = np.mean(centroids, axis=0)  # Simpan centroid di session state

        st.success("Latihan selesai! Centroid telah dihitung.")

elif menu == "Pengujian":
    # Input gambar untuk pengujian
    uploaded_files = st.sidebar.file_uploader("Upload Gambar untuk Pengujian", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Jika file gambar diunggah dan centroid telah ada
    if uploaded_files and 'centroids' in st.session_state:
        images = []

        for uploaded_file in uploaded_files:
            # Baca gambar
            img = Image.open(uploaded_file)
            img = img.resize((256, 256))  # Ubah ukuran gambar
            img_np = np.array(img)  # Konversi ke numpy array
            images.append(img_np)

        # Menampilkan gambar yang telah tersegmentasi
        for idx, image in enumerate(images):
            st.write(f"Gambar {idx + 1}")

            # Tampilkan gambar asli
            st.image(image, caption="Gambar Asli", use_column_width=True)

            # Lakukan segmentasi
            segmented_image, _ = kmeans_manual(image, k_value, st.session_state.centroids)

            # Tampilkan gambar hasil segmentasi
            st.image(segmented_image, caption=f"Gambar Tersegmentasi dengan {k_value} kluster", use_column_width=True)
    else:
        st.warning("Silakan lakukan latihan terlebih dahulu untuk mendapatkan centroid sebelum melakukan pengujian.")
