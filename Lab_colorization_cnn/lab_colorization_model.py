from keras import Input
from keras.layers import Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, UpSampling2D, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import losses, optimizers
import numpy as np
import pickle

RESOLUTION = 256

X = pickle.load(open("X.pickle", "rb"))
X = np.array(X).reshape((-1, RESOLUTION, RESOLUTION, 1))
Y = pickle.load(open("Y.pickle", "rb"))
Y = np.array(Y).reshape((-1, RESOLUTION, RESOLUTION, 2))

depth = 12
epochs = 2000
batch_size = 16



def InstantiateModel(in_):
    encoder_ = Conv2D(32, (3, 3), padding='same', activation="relu", use_bias=False)(in_)
    encoder_ = Conv2D(32, (3, 3), padding='same', activation="relu", strides=(2, 2), use_bias=False)(encoder_)
    encoder_ = BatchNormalization()(encoder_)
    encoder_ = Conv2D(64, (3, 3), padding='same', activation="relu", use_bias=False)(encoder_)
    encoder_ = Conv2D(64, (3, 3), padding='same', activation="relu", strides=(2, 2), use_bias=False)(encoder_)
    encoder_ = BatchNormalization()(encoder_)
    encoder_ = Conv2D(128, (3, 3), padding='same', activation="relu", use_bias=False)(encoder_)
    encoder_ = Conv2D(128, (3, 3), padding='same', activation="relu", use_bias=False)(encoder_)
    encoder_ = Conv2D(128, (3, 3), padding='same', activation="relu", strides=(2, 2), use_bias=False)(encoder_)
    encoder_ = BatchNormalization()(encoder_)

    decoder_ = Conv2D(128, (3, 3), padding='same', activation="relu", use_bias=False)(encoder_)
    decoder_ = Conv2D(128, (3, 3), padding='same', activation="relu", use_bias=False)(decoder_)
    decoder_ = Conv2D(128, (3, 3), padding='same', activation="relu", use_bias=False)(decoder_)
    decoder_ = UpSampling2D((2, 2))(decoder_)
    decoder_ = BatchNormalization()(decoder_)

    decoder_ = Conv2D(64, (3, 3), padding='same', activation="relu", use_bias=False)(decoder_)
    decoder_ = Conv2D(64, (3, 3), padding='same', activation="relu", use_bias=False)(decoder_)
    decoder_ = Conv2D(64, (3, 3), padding='same', activation="relu", use_bias=False)(decoder_)
    decoder_ = UpSampling2D((2, 2))(decoder_)
    decoder_ = BatchNormalization()(decoder_)
    decoder_ = Conv2D(32, (3, 3), padding='same', activation="relu", use_bias=False)(decoder_)
    decoder_ = Conv2D(32, (3, 3), padding='same', activation="relu", use_bias=False)(decoder_)
    decoder_ = Conv2D(32, (3, 3), padding='same', activation="relu", use_bias=False)(decoder_)
    decoder_ = UpSampling2D((2, 2))(decoder_)
    decoder_ = BatchNormalization()(decoder_)

    # fusion = Concatenate()([decoder_, in_])
    #
    # fusion = Conv2D(64, (3,3), padding='same', activation="relu", strides=1)(fusion)
    # fusion = Conv2D(32, (3, 3), padding='same', activation="relu", strides=1)(fusion)

    output = Conv2D(2, 1, activation='tanh', padding='same', strides=1)(decoder_)

    return output


# model = colorizer_model(depth)


Input_Sample = Input(shape=(None, None, 1))
Output_ = InstantiateModel(Input_Sample)
model = Model(inputs=Input_Sample, outputs=Output_)

model.compile(
    optimizer='adam',
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

model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True)

model.save("lab_colorization_cnn_v7.model")
