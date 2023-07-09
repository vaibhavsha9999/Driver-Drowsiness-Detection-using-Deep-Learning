from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
model = load_model(r'model.h5')
mixer.init()
sound= mixer.Sound(r'alarm.wav')
Score = 0


def process_frames():
    global Score
    
    cap = cv2.VideoCapture(0)  # Use appropriate video source (e.g., file path or camera index)

    while True:
        # Read the current frame
        ret, frame = cap.read()
        height,width = frame.shape[0:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces= face_cascade.detectMultiScale(gray, scaleFactor= 1.2, minNeighbors=3)
        eyes= eye_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors=1)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h), color= (255,0,0), thickness=3 )
        for (ex,ey,ew,eh) in eyes:
            #cv2.rectangle(frame,pt1=(ex,ey),pt2=(ex+ew,ey+eh), color= (255,0,0), thickness=3 )
            
            # preprocessing steps
            eye= frame[ey:ey+eh,ex:ex+ew]
            eye= cv2.resize(eye,(80,80))
            eye= eye/255
            eye= eye.reshape(80,80,3)
            eye= np.expand_dims(eye,axis=0)
            # preprocessing is done now model prediction
            prediction = model.predict(eye)

            # if eyes are closed
            if prediction[0][0]>0.15:
                cv2.putText(frame,'closed',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(0,0,0),
                        thickness=1,lineType=cv2.LINE_AA)
                cv2.putText(frame,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(0,0,0),
                        thickness=1,lineType=cv2.LINE_AA)
                Score=Score+1
                if(Score>15):
                    try:
                        sound.play()
                    except:
                        pass
                
            # if eyes are open
            elif prediction[0][1]>0.95:
                cv2.putText(frame,'open',(10,height-40),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(0,0,0),
                        thickness=1,lineType=cv2.LINE_AA)      
                cv2.putText(frame,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(0,0,0),
                        thickness=1,lineType=cv2.LINE_AA)
                Score = Score-2
                if (Score<0):
                    Score=0




        # Convert the processed frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Yield the frame as a byte string
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    # Release the video source when done
    cap.release()



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Replace with your HTML template

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
