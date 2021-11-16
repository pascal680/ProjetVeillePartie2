import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras

model = keras.models.load_model("model/attempt_colorization1.model")
# img = cv2.imread("./00000115.jpg")
img = cv2.imread("../dataset/nature_images/00000013_(6).jpg")

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