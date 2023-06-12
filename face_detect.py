# import cv2 as cv

# img = cv.imread('D:\OpenCv\images\\rutu.jpg')
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('img',img)
# cv.imshow('gray',gray)
# haar_cascade = cv.CascadeClassifier('FACE DETECTION\haar_face.xml')

# face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=7)
# print(f'Number of faces found = {len(face_rect)}')

# for(x,y,w,h) in face_rect:
#     cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
# cv.imshow('detected images',img)
# cv.waitKey(0)


import cv2 as cv

img = cv.imread('images/group.jpg')

img = cv.resize(img,(int(img.shape[1]*0.15),int(img.shape[0]*0.15)),interpolation=cv.INTER_AREA)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
face_haar = cv.CascadeClassifier('FACE DETECTION/haar_face.xml')

haar_rect = face_haar.detectMultiScale(gray,scaleFactor = 1.01,minNeighbors = 9)

print(f'no. of faces found = {len(haar_rect)}')

for (x,y,w,h) in haar_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness = 2)

cv.imshow('detected images',img)
cv.waitKey(0)




"""
In the context of face detection using a Haar cascade classifier, the (x, y) coordinates represent the top-left corner of the bounding box, and w represents the width of the box, while h represents the height.

To calculate the bottom-right corner of the rectangle, we need to add the width (w) to the x-coordinate and the height (h) to the y-coordinate. This gives us the coordinates of the bottom-right corner.

So, (x+w, y+h) represents the point that is w units to the right of the x-coordinate and h units below the y-coordinate, forming the bottom-right corner of the rectangle.

I apologize for the incorrect explanation in my previous response. Thank you for pointing out the mistake, and I appreciate your patience.
"""