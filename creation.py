import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow_hub as hub

st.header("Image class predictor")

def main():
    file_uploaded = st.file_uploader("Choose the file", type=['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(np.array(image))
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)

def predict_class(image):
    classifier_model = tf.keras.models.load_model(r"saved_model/data_wt.hdf5")
    shape = (200, 200, 3)
    model = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape=shape)])
    test_image = image.resize((200, 200))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['buildings', 'forest', 'glacier', 'mountain','sea','street']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    result = 'The image uploaded is: {}'.format(image_class)
    return result

if __name__ == "__main__":
    main()