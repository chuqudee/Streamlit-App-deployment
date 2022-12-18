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

st.image("""omdena.png""")
st.title("Disaster & Non-Disaster Image Classification")
st.header("Please input an image to be classified:")
st.text("Created by Omdena South-Africa Team")

@st.cache(allow_output_mutation=True)

def classifying(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 200, 200, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (200, 200)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction_percentage = model.predict(data)
    prediction=prediction_percentage.round()
    
    return  prediction,prediction_percentage


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
