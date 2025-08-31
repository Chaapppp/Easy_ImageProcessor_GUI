import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# -------------------- Sidebar --------------------
st.sidebar.title("Settings")

input_mode = st.sidebar.radio("Input Source", ["Camera", "Upload", "URL"])

mode = st.sidebar.selectbox(
    "Processing Mode",
    ["Original", "Grayscale", "Gaussian Blur", "Canny Edge", "Binary Threshold"]
)
# -------------------- Initialize session states --------------------
if "ksize" not in st.session_state:
    st.session_state.ksize = 3
if "sigma" not in st.session_state:
    st.session_state.sigma = 1.0
if "low_thr" not in st.session_state:
    st.session_state.low_thr = 100
if "high_thr" not in st.session_state:
    st.session_state.high_thr = 200
if "thr_val" not in st.session_state:
    st.session_state.thr_val = 127

# -------------------- Input --------------------
if "frame" not in st.session_state:
    st.session_state.frame = None

if input_mode == "Camera":
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        img = Image.open(camera_file).convert("RGB")
        st.session_state.frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

elif input_mode == "Upload":
    upload_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if upload_file is not None:
        img = Image.open(upload_file).convert("RGB")
        st.session_state.frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

else:  # URL
    if "url_input" not in st.session_state:
        st.session_state.url_input = ""
    if "fetch_image" not in st.session_state:
        st.session_state.fetch_image = False

    def fetch_from_url():
        st.session_state.fetch_image = True

    url = st.text_input(
        "Enter Image URL",
        value=st.session_state.url_input,
        key="url_input",
        on_change=fetch_from_url
    )

    st.button("Fetch Image", width='stretch', on_click=fetch_from_url)

    if st.session_state.fetch_image and st.session_state.url_input:
        try:
            response = requests.get(st.session_state.url_input)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            st.session_state.frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except:
            st.error("Cannot fetch image from URL.")
        st.session_state.fetch_image = False  # Reset

frame = st.session_state.frame

# -------------------- Processing Settings --------------------
if frame is not None:
    processed = frame.copy()

    if mode == "Original":
        pass

    elif mode == "Grayscale":
        processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    elif mode == "Gaussian Blur":
        col1, col2 = st.sidebar.columns(2)

        # Force odd kernel size
        def update_slider():
            val = st.session_state.ksize_slider
            if val % 2 == 0:
                val += 1
            st.session_state.ksize_slider = val
            st.session_state.ksize_num = val
            st.session_state.ksize = val

        def update_num():
            val = st.session_state.ksize_num
            if val % 2 == 0:
                val += 1
            st.session_state.ksize_num = val
            st.session_state.ksize_slider = val
            st.session_state.ksize = val

        col1.slider(
            "Kernel Size (odd)", 1, 31, st.session_state.ksize,
            step=2, key="ksize_slider", on_change=update_slider
        )
        col2.number_input(
            "Kernel Size (odd)", 1, 31, st.session_state.ksize,
            step=1, key="ksize_num", on_change=update_num
        )

        # Gaussian sigma with slider + number_input
        col3, col4 = st.sidebar.columns(2)

        def update_sigma_slider():
            st.session_state.sigma = st.session_state.sigma_slider

        def update_sigma_num():
            st.session_state.sigma_slider = st.session_state.sigma_num
            st.session_state.sigma = st.session_state.sigma_num

        col3.slider(
            "Gaussian sigma", 0.1, 10.0, st.session_state.sigma, key="sigma_slider", on_change=update_sigma_slider
        )
        col4.number_input(
            "Sigma value", 0.1, 10.0, st.session_state.sigma, key="sigma_num", on_change=update_sigma_num
        )

        processed = cv2.GaussianBlur(
            frame,
            (st.session_state.ksize, st.session_state.ksize),
            st.session_state.sigma
        )

    elif mode == "Canny Edge":
        col1, col2 = st.sidebar.columns(2)
        col3, col4 = st.sidebar.columns(2)

        def update_low():
            st.session_state.low_thr = st.session_state.low_slider

        def update_low_num():
            st.session_state.low_slider = st.session_state.low_num

        def update_high():
            st.session_state.high_thr = st.session_state.high_slider

        def update_high_num():
            st.session_state.high_slider = st.session_state.high_num

        col1.slider("Canny: threshold1", 0, 255, st.session_state.low_thr, step=1, key="low_slider", on_change=update_low)
        col2.number_input("Threshold1", 0, 255, st.session_state.low_thr, step=1, key="low_num", on_change=update_low_num)
        col3.slider("Canny: threshold2", 0, 255, st.session_state.high_thr, step=1, key="high_slider", on_change=update_high)
        col4.number_input("Threshold2", 0, 255, st.session_state.high_thr, step=1, key="high_num", on_change=update_high_num)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = cv2.Canny(gray, st.session_state.low_thr, st.session_state.high_thr)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    elif mode == "Binary Threshold":
        col1, col2 = st.sidebar.columns(2)

        def update_thr():
            st.session_state.thr_val = st.session_state.thr_slider

        def update_thr_num():
            st.session_state.thr_slider = st.session_state.thr_num

        col1.slider("Threshold value", 0, 255, st.session_state.thr_val, step=1, key="thr_slider", on_change=update_thr)
        col2.number_input("Threshold", 0, 255, st.session_state.thr_val, step=1, key="thr_num", on_change=update_thr_num)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, st.session_state.thr_val, 255, cv2.THRESH_BINARY)
        processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # -------------------- Display --------------------
    st.subheader("Original Image")
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.subheader("Original Image RGB Histogram")
    fig_rgb, ax_rgb = plt.subplots()
    colors = ('b', 'g', 'r')
    for i, col in enumerate(colors):
        hist_rgb = cv2.calcHist([frame], [i], None, [256], [0, 256])
        ax_rgb.plot(hist_rgb, color=col)
    ax_rgb.set_xlabel("Pixel Value")
    ax_rgb.set_ylabel("Count")
    st.pyplot(fig_rgb)
    
    st.subheader("Processed Image")
    st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.subheader("Processed Image Histogram")
    gray_hist = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    fig_gray, ax_gray = plt.subplots()
    ax_gray.hist(gray_hist.ravel(), bins=256, range=(0, 256))
    ax_gray.set_xlabel("Pixel value")
    ax_gray.set_ylabel("Count")
    st.pyplot(fig_gray)

else:
    st.info("No image available. Use Camera, Upload, or enter a valid URL.")
