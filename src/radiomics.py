
import numpy as np
import SimpleITK as sitk
import six
import pandas as pd
from radiomics import firstorder, glcm, glrlm, glszm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import os

from joblib import Parallel, delayed

def getRadiomicSetting():
    return {
        "binWidth": 25,
        "resampledPixelSpacing": None,
        "interpolator": sitk.sitkBSpline,
        "label": 1,
    }

def calculateRadiomics(image, radiomics_setting, verbose=False):
    mask = sitk.GetImageFromArray(np.full(image.shape, 1))
    image = sitk.GetImageFromArray(image)
    feature_ls = {}

    # Add radiomic feature extractors here
    # Example for first order features
    if verbose:
        print("Calculating first order features...")
    firstOrderFeatures = firstorder.RadiomicsFirstOrder(image, mask, **radiomics_setting)
    firstOrderFeatures.enableAllFeatures()
    results = firstOrderFeatures.execute()
    for key, val in six.iteritems(results):
        feature_ls[key] = val

    # Show GLCM features
    if verbose:
        print("Calculating GLCM features...")
    glcmFeatures = glcm.RadiomicsGLCM(image, mask, **radiomics_setting)
    glcmFeatures.enableAllFeatures()
    results = glcmFeatures.execute()
    for key, val in six.iteritems(results):
        feature_ls[key] = val

    # Show GLRLM features
    if verbose:
        print("Calculating GLRLM features...")
    glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **radiomics_setting)
    glrlmFeatures.enableAllFeatures()
    results = glrlmFeatures.execute()
    for key, val in six.iteritems(results):
        feature_ls[key] = val

    # Show GLSZM features
    if verbose:
        print("Calculating GLSZM features...")
    glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **radiomics_setting)
    glszmFeatures.enableAllFeatures()
    results = glszmFeatures.execute()
    for key, val in six.iteritems(results):
        feature_ls[key] = val

    return feature_ls

def extract_features_from_images(image_folder, n_jobs=-1, sample_size=None, verbose=False):
    radiomics_setting = getRadiomicSetting()

    def process_image(filename):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            image = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(image)
            features = calculateRadiomics(image_array, radiomics_setting, verbose=verbose)
            return features, filename.replace('.jpg', '')

    list_images = os.listdir(image_folder)
    if sample_size:
        list_images = list_images[:sample_size]

    results = Parallel(n_jobs=n_jobs)(delayed(process_image)(filename) for filename in list_images)

    all_features, image_ids = zip(*results)
    return pd.DataFrame(all_features, index=image_ids)


def create_ml_model(features, labels, model = LinearRegression()):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")

    return model