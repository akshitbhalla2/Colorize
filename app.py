import streamlit as st
import tensorflow as tf
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb, rgb2gray

st.title("Black-and-white image colorization")
st.sidebar.title("Black-and-white image colorization")

st.markdown("This app is for colorizing black-and-white images️")
st.sidebar.markdown("This app is for colorizing black-and-white images️")

def load_model():
    model = tf.keras.models.load_model("color_model")
    return model

model = load_model()

original = io.imread("me.jpg")
gray = io.imread("gray.jpg")
color = io.imread("color.jpg")
st.image([original, gray, color], width=185, caption=["Original", "Gray (Input)", "Colorized (Output)"])

img_list = []

uploaded_file = st.sidebar.file_uploader("Choose a .jpg image", type="jpg")
if uploaded_file is not None:
    original_img = io.imread(uploaded_file)
    img_list.append(original_img)

    target_size = 256

    # Plot gray image (input)
    img = original_img / 255.
    img = resize(img, (target_size, target_size, 3))
    img = rgb2lab(img)
    gray_img = img[:, :, 0]

    img_list.append(rgb2gray(original_img))

    # Make prediction on the input to get output
    gray_img = gray_img.reshape(1, target_size, target_size, -1)
    pred = model.predict(gray_img)
    pred = pred.reshape(target_size, target_size, 2)
    gray_img = gray_img.reshape(target_size, target_size, 1)

    # Plot colorized image (output)
    result = np.zeros((target_size, target_size, 3))
    result[:, :, 0] = gray_img[:, :, 0]
    result[:, :, 1:] = pred * 128
    result = lab2rgb(result)
    img_list.append(result)

    # Display all 3 images
    st.image(img_list, width=185, caption=["Original", "Gray (Input)", "Colorized (Output)"])