# PPE Safety Kit Object Detection using YOLO


## Overview
This project aims to detect PPE safety kits (Personal Protective Equipment) using YOLO (You Only Look Once) object detection algorithm. The model can detect various types of  equipment including 'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone','Safety Vest', 'machinery', 'vehicle' .The target audience for this project includes individuals or organizations who want to ensure the safety of their employees or people working in hazardous environments.


## Dataset
We used the PPE detection dataset available on Roboflow for training our model. The dataset contains images of people wearing PPE equipment in different environments.Dataset is annotated and have classes = `['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone','Safety Vest', 'machinery','vehicle']` . Training set contains 2.6k images , validation set contains 114 images and Testing set contains 82 images. Input dimensions of single image is 640x640. 

## Tech-Stack  : 

- OpenCV 
- Ultralytics 
- torch 
- tochvision 
- Python 3.10
- flask

