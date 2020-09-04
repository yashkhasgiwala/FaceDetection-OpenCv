import cv2

face_cascade=cv2.CascadeClassifier(r'C:\Users\yashk\Documents\OpenCv Detection\haarcascade_frontalface_default.xml')
eyes_cascade=cv2.CascadeClassifier(r'C:\Users\yashk\Documents\OpenCv Detection\haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier(r'C:\Users\yashk\Documents\OpenCv Detection\haarcascade_smile.xml')

def detect(color,gray):
    face=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(color,(int(x),int(y)),(int(x+w),int(y+h)),(255,0,0),2)
        roi_color=color[x:x+w,y:y+h]
        roi_gray=gray[x:x+w,y:y+h]
        eyes=eyes_cascade.detectMultiScale(roi_gray,1.1,22)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(int(ex),int(ey)),(int(ex+ew),int(ey+eh)),(0,255,0),2)
        smile=smile_cascade.detectMultiScale(roi_gray,1.7,25)
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color,(int(sx),int(sy)),(int(sx+sw),int(sy+sh)),(0,0,255),2)    
    return color

video = cv2.VideoCapture(0)
while True:
    _,color=video.read()
    gray=cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)   
    canvas = detect(color,gray)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()     
