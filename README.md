# face-age-gender-detection-system-


Overview

This project is a Django-based web application that performs face detection and demographic prediction (age and gender) using OpenCV DNN models.
It exposes a REST API endpoint that accepts a base64-encoded image and returns detected face coordinates along with predicted age and gender.

Features

Face detection using OpenCV DNN

Gender classification

Age prediction using weighted estimation

Confidence score reporting

JSON API response

Tech Stack

Python

Django

OpenCV (DNN)

NumPy

Pre-trained Caffe models

Due to GitHub file size limitations, the pre-trained Caffe model files 
(.caffemodel) are not included in this repository.

Please download the following models and place them in the project directory:

- age_net.caffemodel
- gender_net.caffemodel
- opencv_face_detector_uint8.pb

You can download them from the official OpenCV repository:
https://github.com/spmallick/learnopencv/tree/master/AgeGender
