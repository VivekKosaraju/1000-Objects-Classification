# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:11:07 2020

@author: kosaraju vivek
"""

import numpy as np
import streamlit as st
import tensorflow as tf
# Keras
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow.keras.backend as K
from werkzeug.utils import secure_filename
import h5py
import os
import io
from PIL import Image, ImageOps

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def models():
    model=load_model('model')
    model._make_predict_function()
    model.summary()
    return model
    
model = models()

def model_predict(img_path, model):
    # Preprocessing the image
    image = Image.open(img_path).convert('RGB')
    size = (299,299)
    image = ImageOps.fit(image, size)
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    preds=model.predict(image)
    label = decode_predictions(preds)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    return label



def main():
    st.set_option('deprecation.showfileUploaderEncoding',False)
    st.title("Image Classification")
    html1 ="""
    <div style="padding:5px">
    <h5 style="color:blue;text-align:right;font-weight:bold;font-style:arial;">created by &copy;Vivek Kosaraju</h5>
    </div>
    """
    st.markdown(html1,unsafe_allow_html=True)
    html2="""
    <h3 style="color:red;text-align:center;">Please upload your Image &#128071;</h3>
    """
    st.markdown(html2,unsafe_allow_html=True)
    image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    if image_file:
        st.image(image_file,caption="uploaded image",width=10,use_column_width=True)
    
    if st.button("Predict"):
        if image_file is None:
            raise Exception("image not uploaded, please refresh page and upload the image")
        with st.spinner("Predicting......"):
            label=model_predict(image_file,model)
            st.write('%s (%.2f%%)' % (label[1], label[2]*100))
         
    hide_streamlit_style ="""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    html_temp3="""
    <p>This application can able to detect 1000+ images.<br> It was developed by &copy; Vivek Kosaraju<br>
        It uses deep learning techniques for image classification. 
    </p>
    """
    if st.button("About"):
        st.markdown(html_temp3,unsafe_allow_html=True)
        
    html_temp4="""
    <h3 style="color:red;text-align:left;">Connect with me &#128108;</h3>
    """
    st.markdown(html_temp4,unsafe_allow_html=True)
    social1="""
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <style>
        .fa {
        padding: 20px;
        font-size: 30px;
        width: 50px;
        text-align: center;
        text-decoration: none;
        margin: 5px 2px;
        }
        .fa-linkedin {
            color: white;
            }
        .fa-instagram {
            color: white;
            }
    </style>
    <a href="https://www.linkedin.com/in/vivek-kosaraju/" class="fa fa-linkedin"></a>
    <a href="https://www.instagram.com/vivek__kosaraju/" class="fa fa-instagram"></a>
    """ 
    
    st.markdown(social1,unsafe_allow_html=True)
    
    
if __name__=='__main__':
    main()
    





    
