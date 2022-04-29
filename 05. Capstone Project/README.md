# Capstone Project - Dog Breed Identification App

[![N|Solid](https://www.python.org/static/community_logos/python-powered-w-70x28.png)](https://www.python.org/)
[![N|Solid](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)](https://scikit-learn.org/stable/)

## Project Objective

Welcome to the dog breed classifier project. This project uses Convolutional Neural Networks (CNNs)! In this project, you will learn how to build a pipeline to process real-world, user-supplied images. Given an image of a dog, your algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

## Project Components

The project can be broken down in three components as follows:
1. CNN Model

A Convolutional Neural Network (CNN) created using transfer learning that can identify dog breed from images. A pre-computed ResNet-50 network was used in the Jupyter Notebook while a Xception network was used in the file model.py, which was later used on the Flask model.

We chose accuracy as the main metric used to measure the perofrmance of the model. The basic constructed CNN which was used as a baseline had an accuracy of 8.4%. The VGG16 CNN using transfer learning improved the accuracy to 45.8%. The final model used in the Jupyter Notebook ResNet-50 CNN model which had an 80.7% accuracy and was the best tested model. Unfortunately I was not able to run that model on my local PC Flask Application without some errors (due to different keras/tensorflow libraries), so I chose to test a different CNN model. The selected model and an InceptionV3 CNN using transfer learning which had a 78.6% accuracy, which was a little worse than the ResNet-50 CNN.

2. Flask Web App

A flask web app which allows the user to upload the chosen file, load it and then tries to predict the dog breen when presented to a dog image, predict which dog resembles the person when presented with ah human image and diplays an error message when presented with an image with cointains neither a dog or a human.

## Instructions:
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

## Lessons Learned / Next Steps

I found this project quite challenging since I have not used CNN very often. It was very fullfilling to see the project run reasonably well, being able to predict a lot of results.

It was also quite fullfilling to be able to improve the proposed face detection algorithm using Face Detection using MTCNN improving both the speed and realiability of human face detection which seemed to be the largest bottleneck in the Udacity proposed solution.

There are a few improvementes which could be done which would add a lot of value to the application such as:
* Experimenting with differente CNN topologies and different classifications heads in order to improve the model.
* Expand the model to correctly classify other species
* Making the model work on multiple animals/people on the same image
* Identifying the person name when he/she is a famous person

## About Me

📈 Financial market professional with the following certifications:
* CGA - Anbima Asset Manager Certification (Certificado de Gestor Anbima)
* CNPI - Apimec Certified Financial Analyst (Certificado Nacional do Profissional de Investimento)

💻 Machine Learning & Programming
* Py Let's Code - Python Data Science course with more than 400 hours (Let's Code)
* Udacity - Data Scientist Nanodegree - _Ongoing_
* Datacamp - _Ongoing_

