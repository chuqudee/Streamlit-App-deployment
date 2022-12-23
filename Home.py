import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# st.image("""omdena.png""")
# st.title("Disaster & Non-Disaster Image Classification")
# st.header("Please input an image to be classified:")
# st.text("Created by Omdena South-Africa Team")
st.set_page_config(page_title='Omdena South Africa Chapter', layout='wide')


@st.cache(allow_output_mutation=True)


def classifying(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 200, 200, 3), dtype=np.float32)
    image = img
    # image sizing
    size = (200, 200)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction_percentage = model.predict(data)
    prediction = prediction_percentage.round()
    return prediction, prediction_percentage


st.markdown(
        """
        <style>
        .block-container.css-18e3th9.egzxvld2 {
        padding-top: 0;
        }
        header.css-vg37xl.e8zbici2 {
        background: none;
        }
        .css-1mr91h5 {
            background-color: transparent;
        }
        span.css-10trblm.e16nr0p30 {
        text-align: center;
        color: #2c39b1;
        }
        .css-1dp5vir.e8zbici1 {
        background-image: linear-gradient(
        90deg, rgb(130 166 192), rgb(74 189 130)
        );
        }
        .css-tw2vp1.e1tzin5v0 {
        gap: 10px;
        }
        .css-50ug3q {
        font-size: 1.2em;
        font-weight: 600;
        color: #2c39b1;
        }
        .row-widget.stSelectbox {
        padding: 10px;
        background: #ffffff;
        border-radius: 7px;
        }
        .row-widget.stRadio {
        padding: 10px;
        background: #ffffff;
        border-radius: 7px;
        }
        label.css-cgyhhy.effi0qh3, span.css-10trblm.e16nr0p30 {
        font-size: 1.1em;
        font-weight: bold;
        font-variant-caps: small-caps;
        border-bottom: 3px solid #F8BE37;
        }
        .css-12w0qpk.e1tzin5v2 {
        background: #d2d2d2;
        border-radius: 8px;
        padding: 5px 10px;
        }
        label.css-18ewatb.e16fv1kl2 {
        font-variant: small-caps;
        font-size: 1em;
        }
        .css-1xarl3l.e16fv1kl1 {
        float: right;
        }
        div[data-testid="stSidebarNav"] li div a {
        margin-left: 1rem;
        padding: 1rem;
        width: 300px;
        border-radius: 0.5rem;
        }
        div[data-testid="stSidebarNav"] li div::focus-visible {
        background-color: rgba(151, 166, 195, 0.15);
        }
        svg.e1fb0mya1.css-fblp2m.ex0cdmw0 {
        width: 2rem;
        height: 2rem;
        }
        .css-11wiv6u {
        background-color: #f1dfc9;
        }
        </style>
        """, unsafe_allow_html=True
    )

st.image("assets/logo-long.png")
st.title("Disaster & Non-Disaster Image Classification")
st.header("Please input an image to be classified:")
st.text("Created by Omdena South-Africa Team")
st.text("VGG16 is a convolution neural network which has been trained on a subset of the 'ImageNet' dataset, a collection of over 14 million images belonging to 22,000 categories.")
st.text("We use the pre-trained model's architecture to create a new dataset from our input images in this approach.")
st.text("We pass our images through VGG16's convolution layers, it gives  an output of feature stack of the detected visual features.")
st.text("The Implementation Process:")
st.text("VGG16 has been imported from the keras.applications")
st.text("Bringing in a new top portion for the model with using the weights from 'Imagenet'")
st.text("The pre-trained convolution layers are freezed")
st.text("The model summary")
st.text("The images have to be first kept in their respective folders with labels for 'Train' and and an unlabelled folder of images was taken for the 'Test' data.")
st.text("The images in the dataset was  processed before using it to classify while passing through the model. We used the 'Image Data Generator' function imported from keras library to do this")
st.text("After getting the data preprocessed, it was fit into the model to analyze the accuracy  and the history is mapped to get the details")
st.text("The vgg16 model gave us a respectable model accuracy")
st.text("With this the vgg16 model was used on an unlabelled set of images to classify it into two categories based on the type of data entered.")
st.text("The dataset in this case being a set of 'Crimes Images' and the 'Disaster Images'.")

uploaded_file = st.file_uploader("Please upload an image file here... (JPEG or PNG)", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded file', use_column_width=True)
    st.write("")
    label,perc = classifying(image, 'disaster_model.h5')
    if label == 1:
        st.write('This is a disaster image!')
    else:
        st.write('This is a non-disaster image')
