# Driver Drowsiness Detection using Deep Learning


This project implements computer vision and deep learning concepts to detect drowsiness of a driver and sound an alarm if drowsy. The goal is to enhance driver safety by monitoring their state and providing timely alerts.

## Technologies Used
- Python
- OpenCV 
- TensorFlow
- Transfer Learning

## Project Overview
- The project uses harcascade for the face and eye detection.
- Using Transfer learning, the learning of InceptionV3 is transferred to use in the case of drowsiness detection.
- For the training purposes [data](https://www.kaggle.com/datasets/kutaykutlu/drowsiness-detection?select=closed_eye) is used.
- Output of model is simple binary classification, i.e. Eye close and Eye Open.

## How to Run the Project
- Install the required dependencies and libraries.
- Run the flask app using
  ```python
  flask --app app run
