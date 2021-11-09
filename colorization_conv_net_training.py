
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

# X = X/255.0 #can use keras.utils.normalize


# building a linear stack of layers with the sequential model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
# convolutional layer
model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')) #Applies filter on the image in squares of 3x3 matrices and adding it all up with an activation function to reinforce patterns
model.add(MaxPooling2D(pool_size=(2,2), padding='same')) #Compress filtered data for image and reinforce patterns

model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')) #Applies filter on the image in squares of 3x3 matrices and adding it all up with an activation function to reinforce patterns
model.add(MaxPooling2D(pool_size=(2,2), padding = 'same')) #Compress filtered data for image and reinforce patterns

model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')) #Applies filter on the image in squares of 3x3 matrices and adding it all up with an activation function to reinforce patterns

model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')) #Applies filter on the image in squares of 3x3 matrices and adding it all up with an activation function to reinforce patterns

model.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')) #Applies filter on the image in squares of 3x3 matrices and adding it all up with an activation function to reinforce patterns

# flatten output of conv
model.add(Flatten()) #Flattens all tensor with all the 

# hidden layer
# output layer
model.add(Conv2D(2, kernel_size=(3,3), padding="same", strides=(1,1))) #Outputs the category in which the image belongs too according to the current state of the CNN

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit(X, Y, batch_size=32,epochs=20, validation_split=0.1)

model.save('colorization_cnn.model')