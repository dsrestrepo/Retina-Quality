# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import random
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from itertools import cycle
import shutil

# Custom CSS to inject via markdown
custom_css = """
<style>
    /* Modify the color of the progress bar */
    .stProgress > div > div > div > div {
        background-color: #ff4a4b;
    }
</style>
"""

# Inject custom CSS with markdown
st.markdown(custom_css, unsafe_allow_html=True)

# Constants:

IMAGES_MIXED = './mixed_dataset/'
DOWNLOAD = False
SHAPE = (224, 224)
LABEL = 'DR_ICDR'
IMAGE_COL = 'image_id'
TEST_SIZE = 0.3
UNDERSAMPLE = False

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

BACKBONE = 'dinov2_base'
MODE = 'fine_tune'
backbone_mode='fine_tune'

HIDDEN = None
num_classes = 3

BATCH_SIZE = 8
NUM_WORKERS = 2

LOSS = 'focal_loss'
OPTIMIZER = 'adam'

# Define your hyperparameters
num_epochs = 15
learning_rate = 5e-6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def degrade_image(image, focus, illumination, alpha_min=0, alpha_max=0, beta_min=0, beta_max=0, kernel_size=5, blur_method="box"):
    """
    Applies blurring and/or illumination changes to an image to simulate real-world
    deterioration due to focus and lighting issues.
    """
    worsened_image = image.copy()
    if focus:
        # Blurring simulates loss of focus or motion.
        if blur_method == 'box':
            # Box Blur: Averages pixels within a kernel size, smoothing the image.
            worsened_image = cv2.blur(worsened_image, (kernel_size, kernel_size))
        elif blur_method == 'median':
            # Median Blur: Replaces each pixel with the median of its neighborhood, reducing noise.
            worsened_image = cv2.medianBlur(worsened_image, kernel_size)
        elif blur_method == 'bilateral':
            # Bilateral Filter: Blurs while preserving edges by considering both spatial and intensity differences.
            d, sigmaColor, sigmaSpace = random.choice([5, 9, 15]), random.choice([50, 75, 100]), random.choice([50, 75, 100])
            worsened_image = cv2.bilateralFilter(worsened_image, d, sigmaColor, sigmaSpace)
        elif blur_method == 'motion':
            # Motion Blur: Simulates the effect of movement during exposure.
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel_motion_blur /= kernel_size
            worsened_image = cv2.filter2D(worsened_image, -1, kernel_motion_blur)

    if illumination:
        # Adjusts image brightness and contrast to simulate various lighting conditions.
        alpha = random.uniform(alpha_min, alpha_max) 
        beta = random.uniform(beta_min, beta_max) 
        worsened_image = cv2.convertScaleAbs(worsened_image, alpha=alpha, beta=beta)

    return worsened_image

def apply_degradation(uploaded_image, degradation_type, alpha_min=0, alpha_max=0, beta_min=0, beta_max=0, kernel_size=5, blur_method="box"):
    # Read the image from an uploaded file instead of a file path
    uploaded_file.seek(0)  # Reset the pointer to the start of the file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    if file_bytes.size > 0:
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        # Handle the case where the file is empty
        print(f"The uploaded file {uploaded_file.name} is empty.")
       
    modified_image = image
    if image is not None:
        if degradation_type == "focus":
            modified_image = degrade_image(image, True, False, alpha_min, alpha_max, beta_min, beta_max, kernel_size, blur_method)
        elif degradation_type == "illumination":
            modified_image = degrade_image(image, False, True, alpha_min, alpha_max, beta_min, beta_max, kernel_size, blur_method)
        elif degradation_type == "both":
            modified_image = degrade_image(image, True, True, alpha_min, alpha_max, beta_min, beta_max, kernel_size, blur_method)
    return modified_image


def degrade_images_in_df(df, uploaded_files, degradation_type, degradation_percentage, out_directory, alpha_min=0, alpha_max=0, beta_min=0, beta_max=0, kernel_size=5, blur_method="box"):
    # Ensure output folder exists
    if os.path.exists(out_directory):
        # Clear the directory by removing it and recreating it
        shutil.rmtree(out_directory)
    
    # Recreate the output directory
    os.makedirs(out_directory, exist_ok=True)

    # Create a progress bar
    progress_bar = st.progress(0)

    # Process each uploaded file only once
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        filename = uploaded_file.name
        
        # Reset the file pointer and read the file only once
        uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            print(f"Error: Couldn't process image {filename}")
            continue
        
        # Check if the corresponding entry exists in the dataframe
        if not df[df['image_id'] == filename[:-4]].empty:
            output_path = os.path.join(out_directory, filename)
            modified_image = image  # Default to original image

            # Decide on degradation based on the specified percentage
            if np.random.rand() < degradation_percentage:
                # Now directly pass the already decoded image to the degradation function
                modified_image = degrade_image(image, "focus" in degradation_type, "illumination" in degradation_type, alpha_min, alpha_max, beta_min, beta_max, kernel_size, blur_method)

            # Save the modified (or original, if no degradation) image
            is_success, buffer = cv2.imencode(".jpg", modified_image)
            if is_success:
                with open(output_path, "wb") as f:
                    f.write(buffer)

        # Update progress bar
        progress_percentage = idx / len(uploaded_files)
        progress_bar.progress(progress_percentage)

    st.success(f"All images have been degraded and saved to the output folder: '{out_directory}'")


##########################################################
#                                                        #
#                                                        #              
#               Frontend Starts HERE!                    #
#                                                        #              
#                                                        #
##########################################################
                
st.title('Degradation Analysis')

# File uploader that allows multiple files
uploaded_files = st.file_uploader("Select files from the folder (use Ctrl/Shift to select multiple files or Ctrl/A to select all)", accept_multiple_files=True)
# File uploader
label_file = st.file_uploader("Choose a label file",  type=['csv'])


# Dropdown for selecting the type of degradation
degradation_types = ['focus', 'illumination','both']  # Include your actual degradation types
additional_condition = None
alpha_min, alpha_max = None, None
beta_min, beta_max = None, None
blur_method = None
kernel_level = None

left_column, right_column = st.columns(2)

with left_column:
    selected_degradation_type = st.selectbox('Select Type of Degradation', degradation_types)

    # Conditional logic based on the selected degradation type
    if selected_degradation_type == 'focus':
        # Additional dropdown for Focus type
        focus_options = ['box', 'median', 'bilateral', 'motion']
        blur_method = st.selectbox('Select Focus Type', focus_options)
        kernel_values = [5, 8, 13, 21, 34, 55, 89]
        kernel_level = st.select_slider('Kernel', options=kernel_values)

    elif selected_degradation_type == 'illumination':
        # Sliders for Illumination type
        illumination_options = ['darker', 'lighter']
        illumination_condition = st.selectbox('Select Focus Type', illumination_options)
        if illumination_condition == 'darker':
            (alpha_min, alpha_max) = st.slider('Alpha', min_value=0.5, max_value=0.9, value=(0.6, 0.7))
            (beta_min, beta_max) = st.slider('Beta', min_value=-100, max_value=-30, value=(-50, -40))
        if illumination_condition == 'lighter':
            (alpha_min, alpha_max) = st.slider('Alpha', min_value=1.1, max_value=3.0, value=(1.8, 2.3))
            (beta_min, beta_max) = st.slider('Beta', min_value=30, max_value=100, value=(50, 70))
    
    elif selected_degradation_type == 'illumination':
        # Additional dropdown for Focus type
        focus_options = ['box', 'median', 'bilateral', 'motion']
        blur_method = st.selectbox('Select Focus Type', focus_options)
        kernel_values = [5, 8, 13, 21, 34, 55, 89]
        kernel_level = st.select_slider('Kernel', options=kernel_values)

        # Sliders for Illumination type
        illumination_options = ['darker', 'lighter']
        illumination_condition = st.selectbox('Select Focus Type', illumination_options)
        if illumination_condition == 'darker':
            (alpha_min, alpha_max) = st.slider('Alpha', min_value=0.5, max_value=0.9, value=(0.6, 0.7))
            (beta_min, beta_max) = st.slider('Beta', min_value=-100, max_value=-30, value=(-50, -40))
        if illumination_condition == 'lighter':
            (alpha_min, alpha_max) = st.slider('Alpha', min_value=1.1, max_value=3.0, value=(1.8, 2.3))
            (beta_min, beta_max) = st.slider('Beta', min_value=30, max_value=100, value=(50, 70))


    percentage_degraded = st.slider('Percentage degraded', 0, 100, 30)

with right_column:
    # Check if any files have been uploaded
    if uploaded_files:
        # Process the first uploaded file
        uploaded_file = uploaded_files[0]  # Use the first uploaded file for preview
        # Read the image from the uploaded file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Ensure the file pointer is reset for any future reads
        uploaded_file.seek(0)

        # Apply degradation to the uploaded image
        if selected_degradation_type == "focus":
            modified_image = degrade_image(image, True, False, alpha_min, alpha_max, beta_min, beta_max, kernel_level, blur_method)
        elif selected_degradation_type == "illumination":
            modified_image = degrade_image(image, False, True, alpha_min, alpha_max, beta_min, beta_max, kernel_level, blur_method)
        elif selected_degradation_type == "both":
            modified_image = degrade_image(image, True, True, alpha_min, alpha_max, beta_min, beta_max, kernel_level, blur_method)
    
        # Convert the modified image from BGR to RGB for display
        modified_image_rgb = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)
        
        # Display the modified image
        st.image(modified_image_rgb, caption="Preview of Modified Image", use_column_width=True)
    else:
        st.write("Please upload an image to see the preview.")

if st.button('Create test dataset'):
    df = pd.read_csv(label_file)
    # Convert into 3 classes:

    # Normal = 0; Non-proliferative = 1, 2, 3; Proliferative = 4
    # Map values to categories
    df[LABEL] = df[LABEL].apply(lambda x: 'Normal' if x == 0 else ('Non-proliferative' if x in [1, 2, 3] else 'Proliferative'))
    
    degrade_images_in_df(df, uploaded_files, selected_degradation_type, percentage_degraded/100, IMAGES_MIXED, alpha_min, alpha_max, beta_min, beta_max, kernel_level, blur_method)
    