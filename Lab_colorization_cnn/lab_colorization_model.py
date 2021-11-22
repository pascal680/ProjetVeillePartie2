import tensorflow as tf
from keras.optimizer_v2.adam import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import losses, optimizers
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pickle
RESOLUTION =256

X = pickle.load(open("X.pickle", "rb"))
X = np.array(X).reshape((-1, RESOLUTION, RESOLUTION, 1))
Y = pickle.load(open("Y.pickle", "rb"))
Y = np.array(Y).reshape((-1, RESOLUTION, RESOLUTION, 2))

depth = 10
epochs = 3
batch_size = 6



def colorizer_model(depth=8):
    # Input
    model = Sequential()
    # input = layers.Input(shape=(RESOLUTION, RESOLUTION, 1))
    print(X.shape[1:])
    model.add(layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', use_bias=False, input_shape= X.shape[1:]))
    # x = layers.Conv2D(filters=64, activation='relu', kernel_size=(3, 3), strides=(1, 1), padding='same',
    #                   use_bias=False)(input)

    # Filters

    for i in range(depth):
        model.add(layers.Conv2D(96, 3, padding='same', activation='relu', use_bias=False))
        model.add(layers.BatchNormalization())
        # x = layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.ReLU()(x)

    model.add(layers.Conv2D(2, 1, padding='same', activation='tanh', use_bias=False))

    return model
    # x = layers.Conv2D(filters=2, activation='tanh', kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x)

model = colorizer_model(depth)

model.compile(
    optimizer= Adam(learning_rate=1E-4),
    metrics=['accuracy'],
    # loss=losses.MeanSquaredError()
    loss= 'mse'
)
print(model.summary())

model.fit(X, Y, batch_size=batch_size,epochs=epochs, validation_split=0.1)

model.save("lab_colorization_cnn.model")