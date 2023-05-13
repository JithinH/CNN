import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input, decode_predictions

model = tf.keras.applications.MobileNetV2(weights="imagenet")

uploaded_file = st.file_uploader("Choose an image file", type="jpg")

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, channels="RGB")

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = mobilenet_v2_preprocess_input(img_array)

    pred_probs = model.predict(img_array)
    predictions = decode_predictions(pred_probs, top=3)[0]

    st.subheader("Top Predictions:")
    for pred in predictions:
        st.write(f"{pred[1]}: {pred[2]*100:.2f}%")
