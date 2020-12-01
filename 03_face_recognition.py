import cv2
import numpy as np
import os

# Path for face image database
#home = "C:/Users/thirat/Documents/git/Face-Recognition-using-Raspberry-Pi/"
home = "/home/linaro/Desktop/Face-Recognition-using-Raspberry-Pi/"
path = home + 'dataset'

#home_haar = "C:/Users/thirat/Documents/git/Face-Recognition-using-Raspberry-Pi/venv/Lib/site-packages/cv2/data/"
home_haar = "/home/pi/opencv_build/opencv/data/haarcascades/"
os.chdir(home_haar)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/pi/FaceRecognition/trainer/trainer.yml')
#recognizer.read(home + 'FaceRecognition/trainer/trainer.yml')
# cascadePath = "/home/pi/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_default.xml"
cascadePath = home_haar + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> KUNAL: id=1,  etc
names = ['None', 'Thirat','Thrumb',"Hikino"]

# Initialize and start realtime video capture
for i in range(10):
    cam = cv2.VideoCapture(i)
    if cam.isOpened():
        break
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)


while True:
    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    max_confidence, max_confidence_label = 0,""
    last_id = "unknown"
    last_x, last_y, last_h = 0,0,0#faces[0][0],faces[0][1],faces[0][3]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        #print(id,confidence)

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence_label = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence_label = "  {0}%".format(round(100 - confidence))
        
        if confidence > max_confidence:
            print(id, confidence, last_id, max_confidence_label)
            max_confidence = confidence
            max_confidence_label = confidence_label
            last_id = id
            last_x, last_y,last_h = x, y,h

        cv2.putText(img, str(last_id), (last_x+5,last_y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(max_confidence_label), (last_x+5,last_y+last_h-5), font, 1, (255,255,0), 1)

    cv2.imshow('camera',img)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
