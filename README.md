# FACE-RECOGNITION-USING-PYTHON
## Introduction
Face Recognition is a simple project developed using Python. 

The project  is developed by using Siamese Neural Network Facial recognition has many benefits in society, including increasing safety and security and preventing crimes. 

It can even help support medical efforts, in some cases.

It is more convenient than the physical database which provides faster access. This way we can recognize any person faster and without any human intervention. 
It uses a webcam to stream frames from a live video to a pre-trained Siamese neural network and using a true image of the user the system is able to authenticate the user in front of the webcam.

We have trained and implemented a robust model that can recognize faces, even when the subject has different expressions and when the photo is taken from different angles.

Program that uses the pre-trained neural network and a webcam, to authenticate the user sitting in front of the computer.

## Facial Recognition Systems
We can break down the face recognition into smaller steps.

Face detection: Detect and isolate faces in the image.

1. We have used a pre-trained cascade classifier for face detection called Haar Cascades.

2. Features with alternating regions of dark and light pixels are known as Haar features.

Face recognition: For each detected face in the image, we run it through a neural network to classify the subject.

## Siamese Neural Networks
The term Siamese means twins. 

When training a Siamese network, 2 or more inputs are encoded and the output features are compared. 

A Siamese network is often shown as two different encoding networks that share weights, but in reality the same network is just used twice before doing backpropagation.

We don't actually need to create two different networks. We only need a single instance of the shared network to be declared in Keras. 

We can create the top and bottom convolutional network using this single instance. 

Because we are reusing this single instance, Keras will automatically understand that the weights are to be shared.


![Picture1](https://user-images.githubusercontent.com/36764949/111611209-3545a180-8802-11eb-8794-a4677d9d8cee.png)!

![Picture2](https://user-images.githubusercontent.com/36764949/111611456-79d13d00-8802-11eb-8e45-0c6ebc076b79.png)


## Future work
Our face recognition system certainly works well under simple conditions. However, it is definitely not fool-proof, and certainly not secure enough to be implemented in important applications. 

For one, the face detection system can be fooled by a static photo. Theoretically, that means we can bypass the authentication by placing a static photo of an authorized user in front of the webcam. 

Techniques to solve this problem are known as anti-spoofing techniques. Anti-spoofing techniques are a keenly studied area in face recognition. In general, there are two main antispoofing techniques used today:

1. Liveness detection: Since a photo is a static two-dimensional image and a real face is dynamic and three-dimensional, we can check for the liveness of the detected face. Ways to perform liveness detection include checking the optic flow of the detected face, and checking the lighting and texture of the detected face in contrast to the surroundings.

2. Machine learning: We can also differentiate a real face from an image by using machine learning! We can train a CNN to classify whether the detected face belongs to a real face or a static image. However, you would need plenty of labeled data (face versus non-face) to accomplish this.


