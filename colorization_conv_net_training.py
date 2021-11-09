
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

X = pickle.load(open("X.pickle", "rb"))
X = np.array(X).reshape(-1, 64, 64, 1)
Y = pickle.load(open("Y.pickle", "rb"))
Y = np.array(Y)

X = X/255.0 #can use keras.utils.normalize


# building a linear stack of layers with the sequential model
model = Sequential()

model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu',input_shape= X.shape[1:]))
# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')) #Applies filter on the image in squares of 3x3 matrices and adding it all up with an activation function to reinforce patterns
model.add(MaxPooling2D(pool_size=(2,2))) #Compress filtered data for image and reinforce patterns
model.add(Dropout(0.25)) #Remove useless filtered images and reduce training time

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')) #Applies filter on the image in squares of 3x3 matrices and adding it all up with an activation function to reinforce patterns
model.add(MaxPooling2D(pool_size=(2,2))) #Compress filtered data for image and reinforce patterns
model.add(Dropout(0.25)) #Remove useless filtered images and reduce training time

# flatten output of conv
model.add(Flatten()) #Flattens all tensor with all the 

# hidden layer
model.add(Dense(400, activation='relu')) #Learning from the reinforced patterns in the convolutional layers
model.add(Dropout(0.4)) #Heavy dropout rate to increase training speed and drop the useless neurons 
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(1, activation='sigmoid')) #Outputs the category in which the image belongs too according to the current state of the CNN

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit(X, Y, batch_size=32,epochs=20, validation_split=0.1)

model.save('colorization_cnn.model')