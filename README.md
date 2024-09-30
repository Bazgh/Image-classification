# Image-classification
Image classification
### Project description
In this Project, we tried pre-trained models such as Yolo and Resnet to classify a dataset of 577 images. although data size is small, these two models do a perfect job on classifying images into four categories.
### dataset 
Our data was collected by taking photos of four different hand gestures including Fist, Open Palm,Peace Sign, and Thums Up
<p align="center">
  <img src="https://github.com/user-attachments/assets/11ade7b0-4f78-4c1a-bfae-19fe04cc10a7" width="200" />
  <img src="https://github.com/user-attachments/assets/7bb2465f-766c-4484-aebb-7855207283c9" width="200" />
  <img src="https://github.com/user-attachments/assets/742404fa-8fa6-4e98-a881-2c2e2fd49301" width="200" />
  <img src="https://github.com/user-attachments/assets/bd25bc74-d1ee-47f9-b109-72e9f2dc55c8" width="200" />
</p>

### Models
1. CNN (Convolutional Neural Network)
   We attempted to design a CNN architecture with various layers aiming at extracting features unique to each class. However, The small size of dataset made even such a strong DL architecture crap at accuracy 28%.
  ## Pre-trained Models
1. YOLO
   YOLO is a Model designed for object detection. It is basically a CNN architecture which detects rectangular boxes within an image. so instead of assigning probabilities to each class, it assigns probabilities to the grids each being a candidate as an object boundary. we use these low to middle features- so called pre trained model to make our algorithm powerful to at least detect the objects and seperate them from back ground.
   The result is an accuracy of 100 precent. However, class Peace Sign is not recognized by this model.
   Below we can see is the result on one validation batch.
   
  <p align="center">
  <img src="https://github.com/user-attachments/assets/6115637c-a65d-4624-9cf1-15c65182023f" width="600" />
</p>

2. RESNET


