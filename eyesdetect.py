import cv2
import numpy as np

def gstreamer_pipeline(
    capture_width=400,
    capture_height=300,
    display_width=400,
    display_height=300,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')

# Create a named window for connections
cv2.namedWindow('Test Eye Detect')

eye_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')

def detect_eyes(img):
    
    face_img = img.copy()
  
    eyes = eye_cascade.detectMultiScale(face_img,scaleFactor=1.2, minNeighbors=5) 
    
    
    for (x,y,w,h) in eyes: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (0,255,0), 3) 
        
    return face_img
    
    
while True: 
    
    ret, frame = cap.read(0) 
     
    frame = detect_eyes(frame)
 
    cv2.imshow('Test Eye Detect', frame) 
 
    c = cv2.waitKey(1) 
    if c == 27: 
        break 
        
cap.release() 
cv2.destroyAllWindows()