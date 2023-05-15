import cv2
import pickle
import face_recognition
import numpy as np
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

# Load the encode file
print("Loading Encode file...")
file = open("EncodeFile.p","rb")
myEncodeListWithID = pickle.load(file)
file.close()
myEncodeList, personID = myEncodeListWithID
print("Encode file loaded")

while True:
    success, img = cap.read()

    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFace = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace, faceloc in zip(encodeCurFace, faceCurFrame):
        matches = face_recognition.compare_faces(myEncodeList,encodeFace)
        faceDis = face_recognition.face_distance(myEncodeList,encodeFace)
        print("matches: ",matches)
        print("Distance: ",faceDis)

        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            print("Known Face Detected")
            print(personID[matchIndex])
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            bbox = x1,y1,x2-x1,y2-y1
            cvzone.cornerRect(img,bbox,rt=0)

    cv2.imshow("WebCam", img)
    cv2.waitKey(1)