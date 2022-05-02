# Capstone Project - Dog Breed Identification App

[![N|Solid](https://www.python.org/static/community_logos/python-powered-w-70x28.png)](https://www.python.org/)
[![N|Solid](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)](https://scikit-learn.org/stable/)

## Problem Introduction

The objective of this project is correctly identify the canine's breed, given an image of a dog. If supplied an image of a human, the code will identify the most resembling dog breed for that particular human.

## Strategy to solve the problem

Since Convolutional Neural Networs (CNN) are used mainly for image processing, classification, segmentation and also for other auto correlated data it seems like the perfect choice to solve this type of problem (breed identification).

We will also need a face detection and a dog detection function which also ended up using some kind of CNN. 

## Metrics

Since we have a balanced enough (target variable varying from 0.04% to 0.12% for each breed) multiclass dataset we chose to use Classification Accuracy as the evaluation metric for the model.

Classification accuracy is the ratio of the number of correct predictions to the total number of predictions made.

## EDA

Since we are dealing with images we are not really able to perform a lot of exploratory data analysis. However we were able to check the distribuition of the target classification (dog breed) variable:

* Train Dataset
mean - 0.75%
std  - 0.18%
min  - 0.39%
max  - 1.15%

* Validation Deataset
mean - 0.75%
std  - 0.16%
min  - 0.48%
max  - 1.08%

* Test Deataset
mean - 0.75%
std  - 0.20%
min  - 0.36%
max  - 1.20%

## Modelling

Inception v3 is a model made up of symmetric and asymmetric building blocks including convolutions, average pooling, max pooling, concatenations, dropouts, and fully connected layers.

The model has shown greater than 78.1% accuracy on the ImageNet dataset.

The final model utilized in the app is a InceptionV3 CNN which used transfer learning. The model uses the the pre-trained Inception V3 model as a fixed feature extractor, where the last convolutional output of Inception V3 is fed as input to our model. We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.

[Advanced Guide to Inception v3](https://cloud.google.com/tpu/docs/inception-v3-advanced)

## Hyperparameter tuning

One of the benefits of CNNs is that they generally have fewer hyperparameters that we need to worry about. The hyperparameters to tune are the number of neurons, activation function, optimizer, learning rate, batch size, and epochs. The chosen model had the hyperparameters used in the Jupyter Notebook, which yielded more than 80% of accuracy.

Reference: 
[Tuning the Hyperparameters and Layers of Neural Network Deep Learning](https://www.analyticsvidhya.com/blog/2021/05/tuning-the-hyperparameters-and-layers-of-neural-network-deep-learning/#:~:text=The%20hyperparameters%20to%20tune%20are,conventional%20algorithms%20do%20not%20have.)

## Results

Both the face and dog detectors run 100% correcty (after imporving face detector using MTCNN).

The benchmark simple CNN had an accuracy of 8.4%. Then we started using transfer learning. Then we started testing different toppologies to try see which one works best for the proposed problem. The first tested topology was a VGG16 CNN which was used as a fixed feature extractor, which was fed as input to our model and managed to improve the results to 45.8%. For the final model we used transfer learning with a ResNet-50 topology which yielded a test accuracy of 80.7%.

OBS: For the final model we had to change the topology to an Inception CNN due to some library version errors.

## Conclusion/Reflection

I found this project quite challenging since I have not used CNN very often. However it was very fullfilling to be able to understand better how CNNs work and getting good results with the final model.

It was also quite fullfilling to be able to improve the proposed face detection algorithm using Face Detection. Using MTCNN I was able to improve noticeably, both the speed and realiability of human face detection which seemed to be the largest bottleneck in the Udacity proposed solution (both by missing some human faces and by breaking down when presented with some images).

## Improvements

There are a few improvementes which could be done which would add a lot of value to the application such as:
* Experimenting with differente CNN topologies and different classifications heads in order to improve the model.
* Expand the model to correctly classify other species
* Making the model work on multiple animals/people on the same image
* Identifying the person name when he/she is a famous person
* Hyperparameter tunig
* Using data augmentation

## Project Objective

Welcome to the dog breed classifier project. This project uses Convolutional Neural Networks (CNNs)! In this project, you will learn how to build a pipeline to process real-world, user-supplied images. Given an image of a dog, your algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

## Project Components

The project can be broken down in three components as follows:
1. CNN Model

A Convolutional Neural Network (CNN) created using transfer learning that can identify dog breed from images. A pre-computed ResNet-50 network was used in the Jupyter Notebook while a Xception network was used in the file model.py, which was later used on the Flask model.

We chose accuracy as the main metric used to measure the perofrmance of the model. The basic constructed CNN which was used as a baseline had an accuracy of 8.4%. The VGG16 CNN using transfer learning improved the accuracy to 45.8%. The final model used in the Jupyter Notebook ResNet-50 CNN model which had an 80.7% accuracy and was the best tested model. Unfortunately I was not able to run that model on my local PC Flask Application without some errors (due to different keras/tensorflow libraries), so I chose to test a different CNN model. The selected model and an InceptionV3 CNN using transfer learning which had a 78.6% accuracy, which was a little worse than the ResNet-50 CNN.

2. Flask Web App

A flask web app which allows the user to upload the chosen file, load it and then tries to predict the dog breen when presented to a dog image, predict which dog resembles the person when presented with ah human image and diplays an error message when presented with an image with cointains neither a dog or a human.

## Instructions

1. Go to the project's root directory

2. Download [DogInceptionV3Data.npz](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) and [DogResNet50Data.npz](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) which contains the pre-computed features for both networks and save them in the ./models folder.

3. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) and the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz) and extract them in the ./data folder.

4. Run the following commands in the project's root directory to set up your database and model.

    a. To run model.py to create the CNN pretrained model and save it
        `python .\models\model.py train_test_model`
        
    b. To load and test the model previously created in a.
        `python .\models\model.py load_trained_model`

5. Run the following command in the app's directory to run your web app.
    `python run.py`

6. Go to http://192.168.0.14:3001
    Select a picture and select submit and have fun!!!

## About Me

📈 Financial market professional with the following certifications:
* CGA - Anbima Asset Manager Certification (Certificado de Gestor Anbima)
* CNPI - Apimec Certified Financial Analyst (Certificado Nacional do Profissional de Investimento)

💻 Machine Learning & Programming
* Py Let's Code - Python Data Science course with more than 400 hours (Let's Code)
* Udacity - Data Scientist Nanodegree - _Ongoing_
* Datacamp - _Ongoing_
