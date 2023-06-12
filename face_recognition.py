import numpy as np
import cv2 as cv
import os
haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = []
for p in os.listdir('D:/openCv/images/PEOPLE'):
    people.append(p)

# feature = np.load('feature.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread('D:/OpenCv/images/PEOPLE/ASUTOSH/ASU1.JPG')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = cv.resize(gray,(int(gray.shape[0]*0.25),int(gray.shape[1]*0.25)),interpolation=cv.INTER_AREA)
# Face recognition 

face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.6,minNeighbors=2)

for (x,y,w,h) in face_rect:
    faces_roi = gray[x:x+w,y:y+h]

    label, confidence = face_recognizer.predict(faces_roi)
    confidence = int(100 * (1 - confidence/300))  # Calculate confidence as a percentage
    print(f'label = {people[label]} with a confidence of {confidence}%')


    cv.putText(img, str(people[label]),(30,30),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('Detected face',img)
cv.waitKey(0)

