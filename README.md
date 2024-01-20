# BRSET: Brazilian Multilabel Ophthalmological Dataset Repository

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


This is a github repository for ophthalmological data quality assesment. The repository contains experiments comparing the performance of good and bad quality images on a classification task. The experiments were performed using the BRSET dataset, which is a Brazilian Multilabel Ophthalmological Dataset. The dataset is publicly available on PhysioNet.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Usage](#usage)
- [Data Analysis](#data-analysis)
- [Quality Assessment](#quality-assessment)
- [Modeling](#modeling)
- [Citation](#citation)
- [License](#license)

## Introduction
In this repo you'll find evaluations using retinal images using deep learning methods, and embedding methods combined with classical machine learning models. 

## Dataset Description
The BRSET dataset is publicly available on PhysioNet, and you can access it through the following DOI link:

- **PhysioNet:** [A Brazilian Multilabel Ophthalmological Dataset (BRSET)](https://doi.org/10.13026/xcxw-8198)

Please refer to the PhysioNet page for detailed information on the dataset structure, contents, and citation guidelines.

## Usage
To use the BRSET dataset and perform technical validation, data analysis, quality assessment, and modeling, you can follow these steps:

1. Clone this repository to your local machine:
```
git clone https://github.com/dsrestrepo/Retina-Quality.git
```

2. Set up your Python environment and install the required libraries by running:

The Python version used here is `Python 3.9`
```
pip install -r requirements.txt
```

3. Explore the dataset and access the data for your analysis.


## Quality Assessment using Deep learning
you'll find the files `modeling_<<task>>.ipynb` where `<<task>>` is the task you want to perform. The tasks are:

* 2 Class Diabetic Retinopathy Classification
* 3 Class Diabetic Retinopathy Classification
* Sex Classification

In all the codes you can custom the hyperparameters and the model architecture and select a backbone.

## Quality Assessment using Embeddings
you'll find the files `embeddings_<<backbone>>.ipynb` where `<<backbone>>` is the backbone used to extract the image embeddings. The files evaluate the quality of the images using 4 different prediction tasks:

* 2 Class Diabetic Retinopathy Classification
* 3 Class Diabetic Retinopathy Classification
* 5 Class Diabetic Retinopathy Classification 
* Sex Classification


## Citation

TODO