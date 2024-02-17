import tensorflow as tf
import cv2
import numpy as np
from cv2 import *
from PIL import Image
import streamlit as st
model = tf.keras.models.load_model('age_gender_classification_model.h5')
# model_Drive_link="https://drive.google.com/file/d/1xR4VfgTQ4OBnU02V7bVIDhMX8g5VUS9M/view?usp=sharing"
IMG_WIDTH, IMG_HEIGHT = 128, 128

def preprocess_image(image_data):
    # Convert image data to NumPy array
    img_array = np.array(image_data)
    # Convert BGR to RGB
    img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # Convert RGB to grayscale
    gray_image = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2GRAY)
    # Resize the image
    resized_image = cv2.resize(gray_image, (IMG_WIDTH, IMG_HEIGHT))
    # Reshape the image for model input
    reshaped_image = resized_image.reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)
    # Normalize the image
    normalized_image = reshaped_image.astype('float32') / 255.0
    return normalized_image



def predict_age_gender(image):
    image = preprocess_image(image)
    gender_pred = 'Female' if model.predict(image)[0][0] == 0 else 'Male'
    age_pred = int(model.predict(image)[1][0])
    return gender_pred,age_pred

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])


if(app_mode=="Home"):
    st.header("Age and Gender Prediction SYSTEM by face image")
    image_path = "home.jpg"
    st.image(image_path)


elif(app_mode=="About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of Male and Female")
    st.code("UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old).\n"
            " The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity.\n"
            " The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc.\n"
            " This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc.")

elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    img = st.camera_input("Take a picture")
    if img is not None:
        test_image = Image.open(img)

    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        # st.snow()
        st.write("Our Prediction")
        gender_pred, age_pred = predict_age_gender(test_image)
        st.text(f"Predicted Gender: {gender_pred} and Age is: {age_pred} ")



