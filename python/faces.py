
import face_recognition_models
import numpy as np 
import cv2 
import os
import numpy as np
from PIL import Image
import pickle

# Define the codec and create VideoWriter object
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainningData.yml")

labels = {"persons_name": 1}
with open("labels.pickle", "rb") as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True): 
	# reads frames from a camera  
	ret, frame = cap.read()  

	# Converts to grayscale space, OCV reads colors as BGR 
	# frame is converted to gray
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)    
	for (x,y,w,h) in faces:
		
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		id_, conf = recognizer.predict(roi_gray)
		if conf >= 45 and conf <= 85:
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
			
		img_item = "image.jpg"
		cv2.imwrite(img_item, roi_gray)
		color = (255, 0, 0)
		stroke = 2
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)


	
	# The original input frame is shown in the window  
	
	cv2.imshow('frame', frame)  
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

# Close the window / Release webcam 
cap.release() 

# After we release our webcam, we also release the out-out.release()  

# De-allocate any associated memory usage  
cv2.destroyAllWindows() 
