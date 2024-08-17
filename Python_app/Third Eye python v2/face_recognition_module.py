import pygame
import os
import cv2
import face_recognition
import numpy as np
import pyttsx3
engine = pyttsx3.init()

pygame.mixer.init()


from firebase import Firebase
#config = {
#  "apiKey": "AIzaSyD1VOTJS6Az-mReE3zpvM_shigqHBjSaYE",
 # "authDomain": "thinkfoteck.firebaseapp.com",
  #"databaseURL": "https://thinkfoteck-default-rtdb.firebaseio.com",
 # "storageBucket": "thinkfoteck.appspot.com"
#}
config = {
  "apiKey": "AIzaSyDuvVwOuFceHGASpCb5v2q5teldykeApkU",
  "authDomain": "http://third-eye-aa2e4.firebaseapp.com",
  "databaseURL": "https://third-eye-aa2e4-default-rtdb.firebaseio.com",
  "storageBucket": "http://third-eye-aa2e4.appspot.com"
}

firebase = Firebase(config)
db = firebase.database()
childName = 'BlindCap2k24'

path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


 
def findEncodings(images):
    encodeList = []
    cnt=0
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        
        encodeList.append(encode)
        print(classNames[cnt])
        print(encode)
        print(len(encode))
        print("ok")
        cnt+=1
        continue
    return encodeList


def face_recognition_fun(img,confirm_count,unknown_count):
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
 
        if matches[matchIndex]:

            confirm_count = confirm_count + 1
            
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            #cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
            #cv2.imwrite("mail_images/known-image.png",img)


            if(confirm_count > 0):

                #welcome_text = "welcome, " + name.lower()
                welcome_text = name.lower() 
                print(name)
                db.child(childName).child('yolo_message').set({'msg' : welcome_text})
                db.child(childName).child('yolo_msgFlag').set('1')
                engine.say(welcome_text)
                engine.runAndWait()

                confirm_count = 0

            return confirm_count,unknown_count
            break

        else:

            confirm_count = 0
            unknown_count = unknown_count + 1

            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            #cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,255),cv2.FILLED)
            cv2.putText(img,'unknown',(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
            #cv2.imwrite("mail_images/unknown-image.png",img)
            print("unknown")
            if(unknown_count > 2):

                #unknown_text = "sorry , you are not allowed"
                #engine.say(unknown_text)
                #engine.runAndWait()

                unknown_count = 0

            return confirm_count,unknown_count
            break

            
 
encodeListKnown = findEncodings(images)
print(type(encodeListKnown))
print('Encoding Complete')
 
