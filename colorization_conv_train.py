import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import losses, optimizers
from matplotlib import pyplot as plt
import numpy as np
import glob
# from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW, SGDW
import cv2
import pickle

from DatasetGenerator import SegmentationDataset

X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

depth = 10
epochs = 20
batch_size = 6


def colorizer_model(depth=8):

    # Input

    input = layers.Input(shape=(256, 256, 1))
    x = layers.Conv2D(filters=64, activation='relu', kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(input)

    # Filters

    for i in range(depth):
        x = layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    # Output

    x = layers.Conv2D(filters=2, activation='sigmoid', kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x)

    print(tf.shape(x))

    print(np.shape(x[:,:,:,1][..., None]))
    u = layers.Lambda(lambda x: x[:,:,:,0][..., None])(x)
    v = layers.Lambda(lambda x: x[:,:,:,1][...,None])(x)

    output = layers.Concatenate()([u, v])

    return Model(inputs=input, outputs=output)

DATADIR = "B:/SCHOOL/AL/sem_9/veille_techno/partie2/dataset/nature_images/train"

train_dataset, val_dataset = SegmentationDataset(DATADIR,
                                                DATADIR,
                                                file_id_regex="(\\d*)",
                                                seed=42,
                                                image_size=(256, 256),
                                                batch_size=5,
                                                augment_dataset=False).get_train_val_datasets()

XX = tf.data.Dataset.from_tensor_slices(X).repeat()
YY = tf.data.Dataset.from_tensor_slices(Y).repeat()

print(tf.__version__, "Version")
model = colorizer_model(depth)

y_img, yuv_img = next(iter(train_dataset))
plt.imshow(y_img[0])
plt.show()
plt.imshow(yuv_img[0])
plt.show()


model.compile(
    optimizer='adam',
    metrics=['accuracy'],
    loss=losses.MeanSquaredError())

print(model.summary())

model.fit(train_dataset,
    steps_per_epoch=len(train_dataset) // 5,
    #validation_split=0.1,
    epochs=5)


#model.load()
img = cv2.imread("./00000115.jpg")
#file = tf.io.read_file("B:\\SCHOOL\\AL\\sem_9\\veille_techno\\partie2\\dataset\\natureimages\\train\\00000114(6).jpg")
#image = tf.image.decode_jpeg(file)

float_image_resized= tf.image.resize(img, size=(256, 256), method="bicubic", antialias=True)

img_yuv = tf.image.rgb_to_yuv(float_image_resized)  #Convertit les images array RGB a des images arrays YUV

float_image = tf.cast(img_yuv, tf.float32) * (1.0 / 255.0)

img_yuv = float_image[0:256, 0:256]

# img_yuv = img_yuv.astype(np.float32)

y = img_yuv[:, :, 0]

img = y.numpy().reshape((1, 256, 256, 1))

result = model.predict(img)

result = tf.cast(result[0, :, :, :], dtype=tf.float32).numpy() * 255.0
result = np.concatenate((y.numpy().reshape((256,256,1)) * 255.0, result), axis=-1)

backtorgb = cv2.cvtColor(result, cv2.COLOR_YUV2RGB)
backtorgb = backtorgb.astype(np.uint8)
cv2.imshow("Capturing", backtorgb)
key = cv2.waitKey(0)