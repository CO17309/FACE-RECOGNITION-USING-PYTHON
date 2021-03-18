# FACE-RECOGNITION-USING-PYTHON
## Introduction
The facial recognition system uses a webcam to stream frames from a live video to a pre-trained Siamese neural network and using a true image of the user the system is able to authenticate the user in front of the webcam.
We have trained and implemented a robust model that can recognize faces, even when the subject has different expressions and when the photo is taken from different angles.
Program that uses the pre-trained neural network and a webcam, to authenticate the user sitting in front of the computer.
##Facial Recognition Systems
We can break down the face recognition into smaller steps.
Face detection: Detect and isolate faces in the image.
1. We have used a pre-trained cascade classifier for face detection called Haar Cascades.
2. Features with alternating regions of dark and light pixels are known as Haar features.
Face recognition: For each detected face in the image, we run it through a neural network to classify the subject.
##Siamese Neural Networks
The term Siamese means twins. 
When training a Siamese network, 2 or more inputs are encoded and the output features are compared. 
A Siamese network is often shown as two different encoding networks that share weights, but in reality the same network is just used twice before doing backpropagation.
We don't actually need to create two different networks. We only need a single instance of the shared network to be declared in Keras. 
We can create the top and bottom convolutional network using this single instance. 
Because we are reusing this single instance, Keras will automatically understand that the weights are to be shared.




