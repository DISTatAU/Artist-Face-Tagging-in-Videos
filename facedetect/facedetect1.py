import numpy as np
import cv2
import os
from pathlib import Path
face_cascade = cv2.CascadeClassifier('I:\\fyp\\lib\\xml\\haarcascade_frontalface_default.xml')
directory_in_str='I:\\fyp\\dataset\\multiple face\\'
newdirectory_in_str='I:\\fyp\\dataset\\newmultiple face\\'
pathlist = Path(directory_in_str).glob('**/*.jpg')
for path in pathlist:
	path_in_str = str(path)
	img = cv2.imread(path_in_str)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	i=0
	for (x,y,w,h) in faces:
		cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		cv2.imshow('img',gray)
		cv2.waitKey(0)
		newpath= os.path.basename(path_in_str)
		#print (newdirectory_in_str ,newpath," \n")
		cv2.imwrite(newdirectory_in_str+str(i)+newpath,roi_gray)
		i=i+1;
		#cv2.waitKey(0)
cv2.destroyAllWindows()
