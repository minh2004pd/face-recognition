import cv2
import face_recognition
import pickle
import os

folderPath = 'images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
personID = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    personID.append(os.path.splitext(path)[0])
print(personID)
print(len(imgList))

def findEncoding(imgList):
    encodeList = []
    for img in imgList:
        cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding started...")
myEncodeList = findEncoding(imgList)
myEncodeListWithID = [myEncodeList,personID]
print("Encoding complete")

file = open("EncodeFile.p","wb")
pickle.dump(myEncodeListWithID,file)
file.close()
print("Save file complete")