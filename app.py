import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Analisis Kematangan Buah Berbasis Warna",
    page_icon="🍎",
    layout="centered"
)

# =========================
# CSS ELEGAN
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #f9fafb, #ffffff);
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    font-weight: 800;
    color: #111827;
}
.subtitle {
    color: #6b7280;
    font-size: 14px;
}
.card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-top: 20px;
}
[data-testid="metric-container"] {
    border-radius: 12px;
    box-shadow: 0 6px 15px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<h1 style="text-align:center;">🍎 Analisis Kematangan Buah</h1>
<p class="subtitle" style="text-align:center;">
Pendekatan clustering warna LAB dan proporsi warna dominan
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Unggah gambar buah", type=["jpg", "jpeg", "png"]
)

# =========================
# UTIL RGB -> LAB (manual)
# =========================
def rgb_to_lab(rgb):
    rgb = rgb / 255.0
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    rgb *= 100

    X = rgb[:,0] * 0.4124 + rgb[:,1] * 0.3576 + rgb[:,2] * 0.1805
    Y = rgb[:,0] * 0.2126 + rgb[:,1] * 0.7152 + rgb[:,2] * 0.0722
    Z = rgb[:,0] * 0.0193 + rgb[:,1] * 0.1192 + rgb[:,2] * 0.9505

    X /= 95.047
    Y /= 100.000
    Z /= 108.883

    f = lambda t: np.where(t > 0.008856, t ** (1/3), (7.787 * t) + (16/116))
    fx, fy, fz = f(X), f(Y), f(Z)

    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.stack([L, a, b], axis=1)

# =========================
# MODEL DETEKSI
# =========================
def deteksi_kematangan(img):
    img = img.convert("RGB")
    data = np.array(img)
    pixels = data.reshape(-1, 3)

    lab = rgb_to_lab(pixels)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(lab)
    centers = kmeans.cluster_centers_

    counts = np.bincount(labels)
    proportions = counts / counts.sum()

    warna = {
        "Hijau": 0,
        "Kuning": 0,
        "Oranye": 0,
        "Merah": 0,
        "Coklat": 0
    }

    for i, center in enumerate(centers):
        L, a, b = center
        if a < -10:
            warna["Hijau"] += proportions[i]
        elif b > 30 and a < 20:
            warna["Kuning"] += proportions[i]
        elif a > 20 and b > 20:
            warna["Oranye"] += proportions[i]
        elif a > 35:
            warna["Merah"] += proportions[i]
        else:
            warna["Coklat"] += proportions[i]

    if warna["Hijau"] > 0.5:
        status = "Masih Mentah 🟢"
    elif warna["Merah"] + warna["Oranye"] > 0.6:
        status = "Matang 🔴"
    else:
        status = "Setengah Matang 🟡"

    return status, warna

# =========================
# OUTPUT
# =========================
if uploaded_file:
    img = Image.open(uploaded_file)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(img, caption="Gambar Buah", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    status, warna = deteksi_kematangan(img)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(status)

    cols = st.columns(len(warna))
    for col, (k, v) in zip(cols, warna.items()):
        col.metric(k, f"{v*100:.1f}%")

    st.caption(
        "Prediksi didasarkan pada komposisi warna dominan "
        "hasil clustering LAB (tanpa OpenCV)."
    )
    st.markdown('</div>', unsafe_allow_html=True)
