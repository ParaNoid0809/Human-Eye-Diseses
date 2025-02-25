import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
from recommendation import cnv, dme, drusen, normal
import tempfile

# Function to handle image prediction using a pre-trained model
def predict_retina_disease(test_image_path):
    # Load the pre-trained model
    model = tf.keras.models.load_model("Trained_Model.keras")
    # Process the image (resize and prepare it for prediction)
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = preprocess_input(x)  # Preprocess image for MobileNetV3 model
    # Make predictions
    predictions = model.predict(x)
    return np.argmax(predictions)  # Return the index of the predicted class

# Sidebar for easy navigation between different pages
st.sidebar.title("Dashboard")
selected_page = st.sidebar.selectbox("Choose a Page", ["Home", "About", "Disease Identification"])

# Home Page
if selected_page == "Home":
    st.markdown("""
    ## Welcome to the Retinal OCT Analysis Platform

#### **What is Optical Coherence Tomography (OCT)?**

OCT is a non-invasive imaging technique that provides high-resolution, cross-sectional images of the retina. It's crucial for the detection and monitoring of retinal diseases like choroidal neovascularization (CNV), diabetic macular edema (DME), and age-related macular degeneration (AMD).

##### **Why OCT Matters**

This platform leverages machine learning to automatically analyze and interpret OCT scans, helping medical professionals identify retinal conditions more quickly and accurately.

---

#### **Features of This Platform:**

- **Automated Disease Classification**: Using advanced AI, we classify OCT scans into four categories: **Normal**, **CNV**, **DME**, and **Drusen**.
- **High-Quality Images**: View detailed images of both healthy retinas and common retinal conditions.
- **Efficient Workflow**: Upload scans, analyze them, and get results fast.

---

#### **Understanding Retinal Diseases in OCT**

- **Choroidal Neovascularization (CNV)**: Characterized by abnormal blood vessels and subretinal fluid.
- **Diabetic Macular Edema (DME)**: Caused by retinal thickening and fluid accumulation.
- **Drusen (Early AMD)**: Small yellow deposits under the retina.
- **Normal Retina**: Healthy retina with no visible abnormalities.

---

#### **Dataset Information**

This platform uses a dataset of **84,495 OCT images** categorized into four main groups: Normal, CNV, DME, and Drusen. These images were sourced from various leading medical centers worldwide, ensuring high-quality data for analysis.

--- 

#### **Get Started**

- **Upload OCT Images**: Upload your OCT scans for analysis.
- **Review Results**: View the classification and detailed insights.
- **Learn More**: Explore the different retinal diseases and how OCT scans help detect them.

--- 

#### **Contact Us**

For questions or assistance, feel free to reach out to our support team via the contact page.
""")

# About the Project Page
elif selected_page == "About":
    st.header("About the Project")
    st.markdown("""
    #### **The Dataset**

OCT is a valuable diagnostic tool in ophthalmology. It provides cross-sectional images of the retina, helping physicians identify diseases like CNV, DME, and Drusen. 

The dataset used in this platform consists of high-resolution OCT images labeled into four categories:
- **CNV** (Choroidal Neovascularization)
- **DME** (Diabetic Macular Edema)
- **Drusen** (Early AMD)
- **Normal**

Images were carefully curated from renowned medical centers, ensuring the accuracy of the labels. They underwent rigorous grading by multiple ophthalmologists, with a final verification process by retinal specialists to ensure the highest level of data integrity.

""")

# Disease Identification Page
elif selected_page == "Disease Identification":
    st.header("Retinal OCT Disease Identification")
    test_image = st.file_uploader("Upload Your OCT Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        # Save the uploaded image to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as tmp_file:
            tmp_file.write(test_image.read())
            temp_file_path = tmp_file.name
    
    if st.button("Predict") and test_image is not None:
        with st.spinner("Analyzing Image..."):
            # Make a prediction based on the uploaded image
            result_index = predict_retina_disease(temp_file_path)
            # Mapping class indices to disease names
            class_names = ['CNV', 'DME', 'Drusen', 'Normal']
        st.success(f"The model predicts this is a: **{class_names[result_index]}**")

        # Provide more information about the prediction
        with st.expander("Learn More About This Condition"):
            if result_index == 0:
                st.write("**Choroidal Neovascularization (CNV)**: Abnormal blood vessels and subretinal fluid.")
                st.image(test_image)
                st.markdown(cnv)
            elif result_index == 1:
                st.write("**Diabetic Macular Edema (DME)**: Retinal thickening with intraretinal fluid.")
                st.image(test_image)
                st.markdown(dme)
            elif result_index == 2:
                st.write("**Drusen (Early AMD)**: Accumulation of small yellow deposits in the retina.")
                st.image(test_image)
                st.markdown(drusen)
            elif result_index == 3:
                st.write("**Normal Retina**: No visible abnormalities.")
                st.image(test_image)
                st.markdown(normal)
