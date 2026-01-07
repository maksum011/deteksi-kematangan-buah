import streamlit as st
import numpy as np
from PIL import Image

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Analisis Kematangan Buah",
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
Pendekatan distribusi warna HSV multi-kategori
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Unggah gambar buah", type=["jpg", "jpeg", "png"]
)

# =========================
# RGB → HSV (NUMPY)
# =========================
def rgb_to_hsv(img):
    img = img / 255.0
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    cmax = np.max(img, axis=2)
    cmin = np.min(img, axis=2)
    delta = cmax - cmin + 1e-6

    hue = np.zeros_like(cmax)

    mask = cmax == r
    hue[mask] = (60 * ((g - b) / delta) % 360)[mask]

    mask = cmax == g
    hue[mask] = (60 * ((b - r) / delta + 2))[mask]

    mask = cmax == b
    hue[mask] = (60 * ((r - g) / delta + 4))[mask]

    sat = delta / (cmax + 1e-6)
    val = cmax

    return hue, sat, val

# =========================
# MODEL DETEKSI MULTI WARNA
# =========================
def deteksi_kematangan(img):
    img = img.convert("RGB")
    data = np.array(img).astype(float)

    hue, sat, val = rgb_to_hsv(data)

    total = hue.size

    warna = {
        "Hijau": np.sum((hue >= 60) & (hue < 140)),
        "Kuning": np.sum((hue >= 40) & (hue < 60)),
        "Oranye": np.sum((hue >= 20) & (hue < 40)),
        "Merah": np.sum((hue < 20) | (hue >= 340)),
        "Coklat/Gelap": np.sum(val < 0.3)
    }

    for k in warna:
        warna[k] /= total

    # LOGIKA KEPUTUSAN (OBYEKTIF)
    if warna["Hijau"] > 0.45:
        status = "Masih Mentah 🟢"
    elif warna["Merah"] + warna["Oranye"] > 0.55:
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
        "Prediksi kematangan berdasarkan distribusi warna HSV "
        "dengan banyak kategori warna dominan."
    )
    st.markdown('</div>', unsafe_allow_html=True)
