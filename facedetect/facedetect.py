import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('I:\\fyp\\lib\\xml\\haarcascade_frontalface_default.xml')
img = cv2.imread('I:\\fyp\\dataset\\faces95\\adhast\\adhast.2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
			
cv2.imshow('img',gray)
cv2.imwrite('adhast11.png',roi_gray)
#cv2.imwrite('adhast12.png',roi_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
