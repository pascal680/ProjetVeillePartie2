import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf

DATADIR = "B:/SCHOOL/AL/sem_9/veille_techno/partie2/dataset/mirflickr"
training_data = []

def create_training_data():
        for img in os.listdir(DATADIR):
            try:
                tf.image.rgb_to_yuv(img)
                img_array = cv2.imread(img)
                print(img_array.shape())
                plt.imshow(img_array)
                plt.show()

                training_data.append(img_array)
            except Exception as e:
                pass


create_training_data()

print(len(training_data))   


X = []
Y = []
