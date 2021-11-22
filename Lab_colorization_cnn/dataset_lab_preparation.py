import os
import cv2
import numpy as np

RESOLUTION = 256

DATADIR = "B:/SCHOOL/AL/sem_9/veille_techno/partie2/dataset/nature_images/train"

def load_images_from_folder(directory, max_count = -1): #Va chercher les images du dataset et les convertit en image array rgb

    images = []

    counter = 0

    for file_name in os.listdir(directory):

        if counter == max_count:    #Permet de limieter la quantite des images a prendre du dataset

            break

            if file_name == train and file_name == test:
                pass

        path = os.path.join(directory, file_name) #Va chercher le path absolut de l'image specifique

        if os.path.exists(path):    #Verifie si l'image au path designer existe

            img = cv2.imread(path)  #Convertit l'image RGB en image array RGB

            if img is not None: #Juste un check pour eviter des problemes

                height, width, channels = img.shape
                if height > RESOLUTION and width > RESOLUTION:

                    images.append(img) #Ajoute la RGB image Array a la liste de retour

                    counter+= 1

    return images   #Retour la liste d"image array de tous les elements du dataset, jusqu'au max count

def rgb_to_lab(list_images):
    x_train = []
    y_train = []

    for img in list_images:
        img_lab = cv2.resize(img, (RESOLUTION, RESOLUTION))
        img_lab = cv2.cvtColor(img_lab, cv2.COLOR_RGB2Lab)
        # print(img_lab.shape)

        # print(img_lab.shape, "resized shape")

        # print(img_lab[:,:,1])

        img_l = img_lab[:, :, 0]

        img_ab = img_lab[:, :, 1:]

        img_ab = img_ab/128 -1

        # print(img_ab[:, :, 1])

        x_train.append(img_l)
        y_train.append(img_ab)

    return x_train, y_train





images = load_images_from_folder(DATADIR, 960)
print(len(images))
x_train, y_train = rgb_to_lab(images)


import pickle  # Utilise pickle pour sauvegarder notre dataset en tant que les liste necessaire pour l'entrainement

pickle_out = open("X.pickle", "wb")

pickle.dump(x_train, pickle_out)

pickle_out.close()


pickle_out = open("Y.pickle", "wb")

pickle.dump(y_train, pickle_out)

pickle_out.close()