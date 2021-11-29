import os
import cv2
import numpy as np

RESOLUTION = 384

DATADIR = "B:/SCHOOL/AL/sem_9/veille_techno/partie2/dataset/nature_images/hand_picked"
patch_size = 256
stride = 32



def data_augmentation(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def generate_patches(file_name):
    img = cv2.imread(file_name)

    h, w = img.shape[:2]

    patches = []

    if h > RESOLUTION and w > RESOLUTION:
        h_c = int((h - RESOLUTION) * 0.5)
        w_c = int((w - RESOLUTION) * 0.5)

        top, left = h_c, w_c
        bottom, right = h_c + RESOLUTION, w_c + RESOLUTION

        img = img[top:bottom, left:right]

        h, w = img.shape[:2]

        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                x = img[i:i + patch_size, j:j + patch_size]

                patches.append(data_augmentation(x, np.random.randint(0, 8)))

    return patches


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

            patches = generate_patches(path)  #Convertit l'image RGB en image array RGB
            counter += 1
            for patch in patches:
                images.append(patch) #Ajoute la RGB image Array a la liste de retour


    return images   #Retour la liste d"image array de tous les elements du dataset, jusqu'au max count


def rgb_to_lab(list_images):
    x_train = []
    y_train = []

    for img in list_images:
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        # print(img_lab.shape)

        # print(img_lab[:,:,1])

        img_l = img_lab[:, :, 0]/ 256

        img_ab = img_lab[:, :, 1:]

        img_ab = img_ab/128 -1

        # print(img_l[:, :])
        # print(np.array(img_l).shape)

        x_train.append(img_l)
        y_train.append(img_ab)

    return x_train, y_train





images = load_images_from_folder(DATADIR, 22)
print(len(images))
x_train, y_train = rgb_to_lab(images)


import pickle  # Utilise pickle pour sauvegarder notre dataset en tant que les liste necessaire pour l'entrainement

pickle_out = open("X.pickle", "wb")

pickle.dump(x_train, pickle_out)

pickle_out.close()


pickle_out = open("Y.pickle", "wb")

pickle.dump(y_train, pickle_out)

pickle_out.close()