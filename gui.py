import cv2
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from PIL import Image

first_name = st.text_input("First name", "Jorji", help="Jorji")
last_name = st.text_input("Last name", "Costava", help="Costava")
date_of_birth = st.date_input("Date of birth", help="1923/11/23")
gender = st.radio("Gender", ("Male", "Female", "Gmail")) # TODO remove joke lol
study_date = st.date_input("Study date", help="1982/11/23")
study_comment = st.text_area("Study comment", help="Ligma")
detectors_slider = st.slider("Detectors", 90, 720, step=90)
scans_slider = st.slider("Scans", 90, 720, step=90)
range_slider = st.slider("Range", 45, 270, step=45)

use_filter = st.sidebar.checkbox("Use filter", False)
show_stats = st.sidebar.checkbox("Show RMSE", False)

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_image is not None:
    image = np.asarray(Image.open(uploaded_image))
    original_image = st.image(image, "Original image")
    sinogramme = st.image(image, "Sinogramme")
    reconstructed_image = st.image(image, "Reconstructed image")
    if use_filter:
        filtered_image = st.image(image, "Filtered image")
    save_dicom = st.button("Save DICOM file")
    if save_dicom:
        # TODO save real DICOM file
        file = open("kek.txt", "w")
        file.write("shrek")
        file.close()
        
        st.write("Saved file")


