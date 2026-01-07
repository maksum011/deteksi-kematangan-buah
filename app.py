import streamlit as st
import numpy as np
from PIL import Image

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Fruit Ripeness Detection",
    page_icon="🍅",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align:center;'>🍅 Fruit Ripeness Detection</h1>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload gambar buah", type=["jpg", "jpeg", "png"]
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
        (1 - abs(h_mean - 0) / 180) * 0.5 +  # kedekatan ke merah
        s_mean * 0.3 +                      # intensitas warna
        v_mean * 0.2                       # kecerahan
    )

    if ripeness_score < 0.35:
        status = "Masih Mentah 🟢"
    elif ripeness_score < 0.6:
        status = "Setengah Matang 🟡"
    else:
        status = "Matang 🔴"

    return status, ripeness_score, {
        "Hue Avg": h_mean,
        "Saturation Avg": s_mean,
        "Brightness Avg": v_mean
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
    c1.metric("Hue Avg", f"{fitur['Hue Avg']:.1f}")
    c2.metric("Saturation", f"{fitur['Saturation Avg']:.2f}")
    c3.metric("Brightness", f"{fitur['Brightness Avg']:.2f}")

    st.caption(
        "Deteksi kematangan menggunakan pendekatan rata-rata HSV "
        "dan skor kematangan berbobot (tanpa OpenCV)."
    )
