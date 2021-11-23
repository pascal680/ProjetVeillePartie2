import tensorflow as tf
from keras import Input
from keras.layers import Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizer_v2.adam import Adam
from numpy import concatenate
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

depth = 12
epochs = 50
batch_size = 10



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





def InstantiateModel(in_):
    model_ = Conv2D(16, (3, 3), padding='same', strides=1)(in_)
    model_ = LeakyReLU()(model_)
    # model_ = Conv2D(64,(3,3), activation='relu',strides=1)(model_)
    model_ = Conv2D(32, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    # model_ = BatchNormalization()(model_)
    model_ = MaxPooling2D(pool_size=(2, 2), padding='same')(model_)

    model_ = Conv2D(64, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    # model_ = BatchNormalization()(model_)
    model_ = MaxPooling2D(pool_size=(2, 2), padding='same')(model_)

    model_ = Conv2D(128, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    # model_ = BatchNormalization()(model_)

    model_ = Conv2D(256, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    # model_ = BatchNormalization()(model_)

    model_ = UpSampling2D((2, 2))(model_)
    model_ = Conv2D(128, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    # model_ = BatchNormalization()(model_)

    model_ = UpSampling2D((2, 2))(model_)
    model_ = Conv2D(64, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    # model_ = BatchNormalization()(model_)

    # concat_ = concatenate([model_, in_])

    model_ = Conv2D(64, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    # model_ = BatchNormalization()(model_)


    model_ = Conv2D(32, (3, 3), padding='same', strides=1)(model_)
    model_ = LeakyReLU()(model_)
    # model_ = BatchNormalization()(model_)

    model_ = Conv2D(2,1, activation='tanh', padding='same', strides=1)(model_)

    return model_

# model = colorizer_model(depth)

Input_Sample = Input(shape=(RESOLUTION, RESOLUTION,1))
Output_ = InstantiateModel(Input_Sample)
model = Model(inputs=Input_Sample, outputs=Output_)


model.compile(
    optimizer= "adam",
    metrics=['accuracy'],
    loss='mse'
)
# model.compile(
#     optimizer= Adam(learning_rate=1E-4),
#     metrics=['accuracy'],
#     # loss=losses.MeanSquaredError()
#     loss= 'mse'
# )
print(model.summary())

model.fit(X, Y, batch_size=batch_size,epochs=epochs, validation_split=0.1, shuffle=True)

model.save("lab_colorization_cnn_v3.model")