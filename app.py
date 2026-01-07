import streamlit as st
import numpy as np
from PIL import Image

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Deteksi Kematangan Buah",
    page_icon="🍅",
    layout="centered"
)

# =========================
# CSS KUSTOM (RINGAN)
# =========================
st.markdown("""
<style>
body {
    background-color: #fafafa;
}

.main {
    padding: 2rem;
}

h1 {
    color: #2c3e50;
    font-weight: 700;
}

.uploadedFile {
    border: 2px dashed #e74c3c;
    border-radius: 10px;
}

.stProgress > div > div > div > div {
    background-color: #e74c3c;
}

[data-testid="metric-container"] {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# =========================
# JUDUL
# =========================
st.markdown(
    "<h1 style='text-align:center;'>🍅 Deteksi Kematangan Buah</h1>"
    "<p style='text-align:center; color:#7f8c8d;'>"
    "Analisis warna berbasis HSV tanpa OpenCV"
    "</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Unggah gambar buah", type=["jpg", "jpeg", "png"]
)

# =========================
# MODEL DETEKSI (HSV SCORE)
# =========================
def deteksi_kematangan_hsv(img):
    img = img.convert("RGB")
    data = np.array(img).astype(float) / 255.0

    r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]

    cmax = np.max(data, axis=2)
    cmin = np.min(data, axis=2)
    delta = cmax - cmin + 1e-6

    hue = np.zeros_like(cmax)

    mask = cmax == r
    hue[mask] = (60 * ((g - b) / delta) % 360)[mask]

    mask = cmax == g
    hue[mask] = (60 * ((b - r) / delta + 2))[mask]

    mask = cmax == b
    hue[mask] = (60 * ((r - g) / delta + 4))[mask]

    saturation = delta / (cmax + 1e-6)
    value = cmax

    # Rata-rata global
    h_mean = np.mean(hue)
    s_mean = np.mean(saturation)
    v_mean = np.mean(value)

    # Skor kematangan berbobot
    ripeness_score = (
        (1 - abs(h_mean - 0) / 180) * 0.5 +
        s_mean * 0.3 +
        v_mean * 0.2
    )

    if ripeness_score < 0.35:
        status = "Masih Mentah 🟢"
    elif ripeness_score < 0.6:
        status = "Setengah Matang 🟡"
    else:
        status = "Matang 🔴"

    return status, ripeness_score, {
        "Hue Rata-rata": h_mean,
        "Saturasi Rata-rata": s_mean,
        "Kecerahan Rata-rata": v_mean
    }

# =========================
# OUTPUT APLIKASI
# =========================
if uploaded_file:
    img = Image.open(uploaded_file)

    st.image(img, caption="Gambar Buah", use_container_width=True)

    status, tingkat, fitur = deteksi_kematangan_hsv(img)

    st.subheader(status)
    st.progress(float(min(tingkat, 1.0)))

    c1, c2, c3 = st.columns(3)
    c1.metric("Hue", f"{fitur['Hue Rata-rata']:.1f}")
    c2.metric("Saturasi", f"{fitur['Saturasi Rata-rata']:.2f}")
    c3.metric("Kecerahan", f"{fitur['Kecerahan Rata-rata']:.2f}")

    st.caption(
        "Deteksi kematangan buah menggunakan pendekatan "
        "rata-rata HSV dan skor kematangan berbobot."
    )
