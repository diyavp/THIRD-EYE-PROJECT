import pytesseract
import numpy as np
import cv2

import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 160)

from firebase import Firebase
#config = {
 # "apiKey": "AIzaSyD1VOTJS6Az-mReE3zpvM_shigqHBjSaYE",
 # "authDomain": "thinkfoteck.firebaseapp.com",
 # "databaseURL": "https://thinkfoteck-default-rtdb.firebaseio.com",
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

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
#tessdata_dir_config = r'--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata-main/"'


def ocr_fun(cam):
    _, frame = cam.read()
    cv2.imwrite('Text_Image.png', frame)

    img = cv2.imread('Text_Image.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernl = np.ones((1,1),np.uint8)
    img_dlt = cv2.dilate(img,kernl,iterations=1)
    img_dlt_erd = cv2.erode(img,kernl,iterations=1)
    target = pytesseract.image_to_string(img_dlt_erd)
    print(target)
    target = target.replace("\n", "\t")
    db.child(childName).child('yolo_message').set({'msg' : target})
    db.child(childName).child('yolo_msgFlag').set('1')
    talk(target)
    

def talk(text):

    engine.say(text)
    engine.runAndWait()
