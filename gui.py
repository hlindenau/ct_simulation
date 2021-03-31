import cv2
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from PIL import Image
from main import radon_transform, inv_radon, save_dicom, convert_to_cv2, rmse, filter_img
import datetime
import pydicom

f_name = "Jorji"
l_name = "Costava"
dob = datetime.datetime(*[1963, 11, 23])
st_date = datetime.datetime(*[1991, 12, 25])
st_comment = "Ligma"
sex = "Gmail"

dicom_read = False
dicom_img = None

dicom = st.file_uploader("Read DICOM", "dcm")
if dicom is not None:
    dicom_read = True

    dicom_dataset = pydicom.read_file(dicom)
    f_name = str(dicom_dataset.PatientName).split()[0]
    l_name = str(dicom_dataset.PatientName).split()[1]
    dob = datetime.datetime.strptime(str(dicom_dataset.PatientBirthDate), "%Y/%m/%d %H:%M:%S.%f").date()
    sex = str(dicom_dataset.PatientSex)
    st_date = datetime.datetime.strptime(str(dicom_dataset.ContentDate), "%Y/%m/%d %H:%M:%S.%f").date()
    w = dicom_dataset.Rows
    h = dicom_dataset.Columns
    dicom_img = dicom_dataset.pixel_array

sex_tuple = ("Male", "Female", "Gmail")

first_name = st.text_input("First name", value=f_name, help="Jorji")
last_name = st.text_input("Last name", value=l_name, help="Costava")
date_of_birth = st.date_input("Date of birth", value=dob, help="1963/11/22")
gender = st.radio("Gender", sex_tuple, sex_tuple.index(sex))  # TODO remove joke lol
study_date = st.date_input("Study date", help="1963/11/23", value=st_date)
study_comment = st.text_area("Study comment", value=st_comment, help="Ligma")
detectors_slider = st.slider("Detectors", 90, 1080, step=90, value=360)
scans_slider = st.slider("Scans", 90, 1080, step=90, value=360)
# range_slider = st.slider("Range", 45, 270, step=45)
range_step_slider = st.slider("Range step", .1, 90., step=.1, value=1.)
filter_size = st.slider("Filter size", 3, 100, step=1, value=25)

use_filter = st.sidebar.checkbox("Use filter", False)
show_stats = st.sidebar.checkbox("Show RMSE", False)

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

original_img = None
original_img_norm = None
sinogramme = None
sinogramme_src = None
sinogramme_filtered = None
sinogramme_filtered_norm = None
reconstr_img = None
reconstr_img_norm = None

if dicom_read:
    dicom_img = cv2.normalize(dicom_img, dicom_img, 0, 255, cv2.NORM_MINMAX)
    st.image(dicom_img, "Reconstructed image", clamp=True)
elif uploaded_image is not None:
    img_raw = Image.open(uploaded_image)
    original_img = convert_to_cv2(img_raw)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    st.image(original_img, "Original image")

    sinogramme = radon_transform(original_img, detectors_slider, scans_slider, range_step_slider, 180 / range_step_slider)
    # sinogramme_img_resized = cv2.resize(sinogramme_img, image.shape) # FIXME
    st.image(sinogramme, "Sinogramme")
    sinogramme_src = sinogramme

    if use_filter:
        sinogramme_filtered = filter_img(sinogramme, filter_size)

        sinogramme_filtered_norm = np.zeros(sinogramme_filtered.shape)
        sinogramme_filtered_norm = cv2.normalize(sinogramme_filtered, sinogramme_filtered_norm, 0, 1, cv2.NORM_MINMAX)
        st.image(sinogramme_filtered_norm, "Filtered sinogramme", clamp=True)

        sinogramme_src = sinogramme_filtered

    reconstr_img = inv_radon(sinogramme_src, original_img.shape, detectors_slider, scans_slider, range_step_slider,
                             180 / range_step_slider)
    reconstr_img_norm = np.zeros(original_img.shape)
    reconstr_img_norm = cv2.normalize(reconstr_img, reconstr_img_norm, 0, 1, cv2.NORM_MINMAX)
    st.image(reconstr_img_norm, "Reconstructed image", clamp=True)

    original_img_norm = np.zeros(original_img.shape)
    original_img_norm = cv2.normalize(original_img, original_img_norm, 0, 1, cv2.NORM_MINMAX)

if show_stats and reconstr_img_norm is not None:
    st.write("RMSE: {rmse:.2f}".format(rmse=rmse(original_img_norm, reconstr_img_norm) * 255.))

filename = st.text_input("File name", "jorji", help="Jorji")
save_file = st.button("Save DICOM file")

if save_file:
    save_dicom(filename, first_name + " " + last_name, reconstr_img, study_comment, date_of_birth, gender, study_date)


