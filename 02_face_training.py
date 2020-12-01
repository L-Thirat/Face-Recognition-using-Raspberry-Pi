import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
home = "C:/Users/thirat/Documents/git/Face-Recognition-using-Raspberry-Pi/"
# home = "/home/pi/"
path = home + 'dataset'

home_haar = "C:/Users/thirat/Documents/git/Face-Recognition-using-Raspberry-Pi/venv/Lib/site-packages/cv2/data/"
# home_haar = "/home/pi/opencv-3.4.1/data/haarcascades/"
os.chdir(home_haar)

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(home_haar + "haarcascade_frontalface_default.xml")


# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return faceSamples, ids


print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
reg_path = home + "FaceRecognition/trainer/trainer.yml"
# recognizer.write(reg_path)
recognizer.save(reg_path)  # worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
