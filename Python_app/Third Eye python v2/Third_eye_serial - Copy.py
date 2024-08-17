from OCR_module import *
from yolo_module import *
from face_recognition_module import *

confirm_count = 0
unknown_count = 0

import cv2
import serial

#ser = serial.Serial('COM5',9600, timeout=0.1)

#cam = cv2.VideoCapture('http://192.168.29.243:8080/video')
cam = cv2.VideoCapture(0)

while True:

    #char = ser.read()
    #char = char.decode("utf-8")

    _, img = cam.read()
    cv2.imshow('Frame', img)
    char = '1'
    if(char=='1'):
        yolo_fun(cam,char)
        try:
            confirm_count,unknown_count=face_recognition_fun(img,confirm_count,unknown_count)
        except:
            print("TypeError")
    char='2'
    if(char=='2'):
        ocr_fun(cam)

    if(cv2.waitKey(1) == ord('q')):
        break

cam.release()        
cv2.destroyAllWindows()
