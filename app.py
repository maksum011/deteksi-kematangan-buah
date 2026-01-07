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
# CSS MEWAH & PROFESIONAL
# =========================
st.markdown("""
<style>
/* Background utama */
body {
    background: linear-gradient(135deg, #f4f6f8, #ffffff);
    font-family: 'Segoe UI', sans-serif;
}

/* Container utama */
.main {
    padding: 2.5rem;
}

/* Judul */
h1 {
    color: #1f2937;
    font-weight: 800;
    letter-spacing: 1px;
}

/* Subjudul */
.subtitle {
    color: #6b7280;
    font-size: 15px;
    margin-top: -10px;
}

/* Card efek mewah */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
    margin-top: 20px;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #dc2626, #f97316);
    border-radius: 10px;
}

/* Metric */
[data-testid="metric-container"] {
    background: #ffffff;
    border-radius: 14px;
    padding: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
}

/* Upload box */
section[data-testid="stFileUploader"] {
    border: 2px dashed #d1d5db;
    padding: 15px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown(
    """
    <h1 style="text-align:center;">🍅 Deteksi Kematangan Buah</h1>
    <p class="subtitle" style="text-align:center;">
    Analisis warna berbasis HSV dengan skor kematangan berbobot
    </p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Unggah gambar buah", type=["jpg", "jpeg", "png"]
)

# =========================
# MODEL DETEKSI HSV
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

    h_mean = np.mean(hue)
    s_mean = np.mean(saturation)
    v_mean = np.mean(value)

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

    return status, ripeness_score, h_mean, s_mean, v_mean

# =========================
# OUTPUT
# =========================
if uploaded_file:
    img = Image.open(uploaded_file)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(img, caption="Gambar Buah", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    status, score, h, s, v = deteksi_kematangan_hsv(img)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(status)
    st.progress(float(min(score, 1.0)))

    c1, c2, c3 = st.columns(3)
    c1.metric("Hue Rata-rata", f"{h:.1f}")
    c2.metric("Saturasi", f"{s:.2f}")
    c3.metric("Kecerahan", f"{v:.2f}")

    st.caption(
        "Pendekatan ini menilai kematangan buah berdasarkan "
        "karakteristik warna global menggunakan ruang warna HSV."
    )
    st.markdown('</div>', unsafe_allow_html=True)
