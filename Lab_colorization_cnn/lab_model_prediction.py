import cv2
import numpy
import tensorflow as tf
import numpy as np
from tensorflow import keras

model = keras.models.load_model("lab_colorization_cnn_v3.model")
# img = cv2.imread("./00000115.jpg")
img = cv2.imread("../../dataset/nature_images/00000013_(6).jpg")

#file = tf.io.read_file("B:\\SCHOOL\\AL\\sem_9\\veille_techno\\partie2\\dataset\\natureimages\\train\\00000114(6).jpg")
#image = tf.image.decode_jpeg(file)

img_lab = cv2.resize(img, (256, 256))
img_lab = cv2.cvtColor(img_lab, cv2.COLOR_RGB2Lab)

img_lab = np.array(img_lab) / 255

img_l = img_lab[:, :, 0]

# img_ab = img_lab[:, :, 1:]

# img_ab = img_ab / 128 - 1

# img_yuv = img_yuv.astype(np.float32)

# l = img_lab[:, :, 0]

img = img_l[np.newaxis,:,:,np.newaxis]

result = model.predict(img)

result = (tf.cast(result[0, :, :, :], dtype=tf.float32).numpy()+1) * 128
result = np.concatenate((img_l[:,:,np.newaxis] * 256.0, result), axis=-1)

print(result.shape)

backtorgb = result.astype(np.uint8)
backtorgb = cv2.cvtColor(backtorgb, cv2.COLOR_LAB2BGR)

cv2.imshow("Capturing", backtorgb)
key = cv2.waitKey(0)