import os
import cv2

DATADIR = "B:/SCHOOL/AL/sem_9/veille_techno/partie2/dataset/mirflickr"

def load_images_from_folder(directory, max_count = -1): #Va chercher les images du dataset et les convertit en image array rgb

    images = []

    counter = 0

    for file_name in os.listdir(directory): 

        if counter == max_count:    #Permet de limieter la quantite des images a prendre du dataset

            break

        path = os.path.join(directory, file_name) #Va chercher le path absolut de l'image specifique

        if os.path.exists(path):    #Verifie si l'image au path designer existe

            img = cv2.imread(path)  #Convertit l'image RGB en image array RGB

            if img is not None: #Juste un check pour eviter des problemes

                images.append(img) #Ajoute la RGB image Array a la liste de retour

                counter+= 1

    return images   #Retour la liste d"image array de tous les elements du dataset, jusqu'au max count




def create_training_data(images):   #Convertit la liste des images array au liste necessaire pour l'entrainement avec un color range YUV

    x_train = [] #La liste des img array utilise pour l'entrainement

    y_train = [] #La liste des valeurs associe avec l'entrainement, agissant comme les labels pour l'entrainement supervise

    for img in images:

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  #Convertit les images array RGB a des images arrays YUV

        y, u, v = cv2.split(img_yuv)    #Separe les colors channel YUV en valeur separe

        x_train.append(y) #Ajoute la color channel de luminosite au tableau de valeur input pour l'entrainement

        y_train.append(cv2.merge((u,v))) #Ajoute les autres color channel, ceux qui definisse la couleur dans le tableau de label

    return x_train, y_train #Retourne les deux listes necessaire pour l'entrainement






dataset = load_images_from_folder(DATADIR, 10) #Va chercher les images array RGB et les garde dans une variable dataset

x_train , y_train = create_training_data(dataset) #Va chercher les donnes necessaire pour l'entrainement machine avec les images convertit en YUV

cv2.imshow("Y", x_train[5]) #Montre le resultat du color Y de l'image specifique

u, v = cv2.split(y_train[5]) #Separe le y_train qui contient les valeurs U et V en les color channel U et V respective

cv2.imshow("u", u) #Montre le resultat du color U de l'image specifique

cv2.imshow("V", v) #Montre le resultat du color V de l'image specifique


cv2.waitKey(0) #Permet de continuer le script sur la fermeture des onglets imshow


print(len(dataset))   #Imprime la longueur du dataset pour verifier

print(len(x_train))   #Imprime la longueur du x_train pour verifier

print(len(y_train))   #Imprime la longueur du y_train pour verifier




import pickle  #Utilise pickle pour sauvegarder notre dataset en tant que les liste necessaire pour l'entrainement

pickle_out = open("X.pickle", "wb")

pickle.dump(x_train, pickle_out)

pickle_out.close()


pickle_out = open("Y.pickle", "wb")

pickle.dump(y_train, pickle_out)

pickle_out.close()
