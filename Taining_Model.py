import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('C:/Users/Shudhanshu Sharma/PycharmProjects/Minor_Project/haarcascade_frontalface_default.xml')

def getImages_and_Labels(path):
    Image_Paths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    IDs = []
    for Image_Path in Image_Paths:
        pilImage = Image.open(Image_Path).convert('L')
        ImgNP = np.array(pilImage,'uint8')

        ID=int(os.path.split(Image_Path)[-1].split(".")[1])
        faces = detector.detectMultiScale(ImgNP)

        for (x,y,w,h) in faces:
            faceSamples.append(ImgNP[y:y+h, x:x+w])
            IDs.append(ID)

    return faceSamples, IDs

faces,IDs = getImages_and_Labels('C:/Users/Shudhanshu Sharma/PycharmProjects/Minor_Project/dataset/')
s = recognizer.train(faces, np.array(IDs))
print("Successfully trained")
recognizer.write('C:/Users/Shudhanshu Sharma/PycharmProjects/Minor_Project/trainer/trainer.yml')