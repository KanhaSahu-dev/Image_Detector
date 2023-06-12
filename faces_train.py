import os
import cv2 as cv
import numpy as np

people = []
for i in os.listdir('D:/openCv/images/people'):
    people.append(i)

dir = r'D:/openCv/images/people'
haar_cascade = cv.CascadeClassifier('haar_face.xml')
feature = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(dir,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            img_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)

            for (x,y,w,h) in img_rect:
                faces_roi = gray[x:x+w,y:y+h]
                feature.append(faces_roi)
                labels.append(label)
                
create_train()
print(f'length of the features list = {len(feature)}')
print(f'length of the labels list = {len(labels)}')

feature = np.array(feature,dtype=object)
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(feature, labels)


print('training done')


face_recognizer.save('face_trained.yml')
np.save('feature.npy',feature)
np.save('labels.npy',labels)

