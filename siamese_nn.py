# Code for training a Siamese neural network
import helper # Module containing few function
import numpy as np # For working with arrays
from keras.layers import Input, Lambda # Input layer to instantiate a Keras tensor 
                                       # and Lambda layer for euclidian distance computation
from keras.models import Model # Group layers into an object

faces_dir = 'att_faces/'

# Import Training and Testing Data
(X_train, Y_train), (X_test, Y_test) = helper.get_data(faces_dir) # To load the respective raw images into NumPy arrays
num_classes = len(np.unique(Y_train)) # Find the unique elements of an array

# Create Siamese Neural Network

# Create a single instance of the shared network
input_shape = X_train.shape[1:] # To describe the dimensions of input
shared_network = helper.create_shared_network(input_shape) # Function to create a Siamese neural network

# Specify the input for the top and bottom layers using the Input class
input_top = Input(shape=input_shape)
input_bottom = Input(shape=input_shape)

# Stack the shared network to the right of the input layers, using the functional method in Keras
output_top = shared_network(input_top)
output_bottom = shared_network(input_bottom)

# Wrap euclidean_distance function inside a Lambda layer
distance = Lambda(helper.euclidean_distance, output_shape=(1,))([output_top, output_bottom])

# combine the distance layer defined in the previous line with our inputs to complete our model
model = Model(inputs=[input_top, input_bottom], outputs=distance)

# Train the model

# Generate negative and positive pairs of images and their label for training a Siamese neural network
training_pairs, training_labels = helper.create_pairs(X_train, Y_train, num_classes=num_classes) 

# Define the parameters of the training
model.compile(loss=helper.contrastive_loss, optimizer='adam', metrics=[helper.accuracy])

# train our model for 10 epochs by calling the fit function
model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_labels,
          batch_size=128,
          epochs=10)

# Save the model
model.save('siamese_nn.h5')

# Verify the structure of our model by calling the summary function
print(model.summary())
