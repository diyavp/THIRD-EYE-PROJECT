import cv2
import numpy as np
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 160)


from firebase import Firebase
#config = {
#  "apiKey": "AIzaSyD1VOTJS6Az-mReE3zpvM_shigqHBjSaYE",
#  "authDomain": "thinkfoteck.firebaseapp.com",
#  "databaseURL": "https://thinkfoteck-default-rtdb.firebaseio.com",
#  "storageBucket": "thinkfoteck.appspot.com"
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

def talk(text):
    engine.say(text)
    engine.runAndWait()

talk('please wait, camera is warming up')
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    #print(output_layers)

    return output_layers


def draw_prediction(img,COLORS, classes, class_id, confidence, x, y, x_plus_w, y_plus_h, char):

    label = str(classes[class_id])
    color = COLORS[class_id]

    print(label)

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow('object detection', img)
    if(cv2.waitKey(1) == ord('q')):
        pass

    if(char == '1'):
        print('Talk Enabled')
        db.child(childName).child('yolo_message').set({'msg' : label})
        db.child(childName).child('yolo_msgFlag').set('1')
        talk(label)

def yolo_fun(cam,char):
    _,image = cam.read()

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open('yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    #print(len(classes))
        
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    flag_bike = 0
    flag_person = 0

    for out in outs:
        for detection in out:
            #print(len(detection))
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

                
        cv2.imshow('object detection', image)


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:

            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

           
    
            draw_prediction(image,COLORS,classes, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), char)

            
            cv2.imshow('object detection', image)

    cv2.imshow('object detection', image)
    
    
