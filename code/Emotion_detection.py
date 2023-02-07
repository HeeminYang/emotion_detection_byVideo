import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Importing required packages
from keras.models import load_model
#import tensorflow as tf
import numpy as np
import argparse
import dlib
import cv2
import json
import pandas as pd
import sys
from pathlib import Path


ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

emotion_offsets = (20, 40)
emotions = {
    0: {
        "emotion": "Angry",
        "color": (193, 69, 42)
    },
    1: {
        "emotion": "Disgust",
        "color": (164, 175, 49)
    },
    2: {
        "emotion": "Fear",
        "color": (40, 52, 155)
    },
    3: {
        "emotion": "Happy",
        "color": (23, 164, 28)
    },
    4: {
        "emotion": "Sad",
        "color": (164, 93, 23)
    },
    5: {
        "emotion": "Suprise",
        "color": (218, 229, 97)
    },
    6: {
        "emotion": "Neutral",
        "color": (108, 72, 200)
    }
}

def shapePoints(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rectPoints(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


faceLandmarks = "./faceDetection/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

emotionModelPath = './models/emotionModel.hdf5'  # fer2013_mini_XCEPTION.110-0.65
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]

# Bring the '.mp4' file list in the folder and put it in videocapture\

root_path = Path('./data/DuckEEsStim/')
file_paths = np.array(list(root_path.rglob('*.m4v')))
print(root_path)
for file_path in file_paths:
    cap = cv2.VideoCapture(file_path.as_posix())
    
    totalcount = 0
    landcount = 0
    sll = []
    
    c = 0
    while True:
        ret, frame = cap.read()
    
        if frame is None:
            break
        try: 
            frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(e)
        #print(f"Count {c} | Reading frame succeed : {frame.shape}", end = ' | ')

        totalcount += 1
        cv2.putText(frame, 'Picture: '+str(totalcount), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #print('Putting text succeed', end = ' | ')
        # if not ret:
        #     break

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print('Converting color succeed', end = ' | ')
        rects = detector(grayFrame, 0)
        #print('Detecting face succeed')
        c += 1
        for rect in rects:
            shape = predictor(grayFrame, rect)
            points = shapePoints(shape)
            
            landcount += 1
            cv2.putText(frame, 'Landmark: ' + str(landcount), (0,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            shape_list = []

            shape_list.append(totalcount)
            shape_list.append(landcount)
                ## append (x, y) in shape_list
            for p in shape.parts():
                shape_list.append(p.x)
                shape_list.append(p.y)
                cv2.circle(frame, (p.x, p.y), 2, (0, 255, 0), -1)        
            
            (x, y, w, h) = rectPoints(rect)
            grayFace = grayFrame[y:y + h, x:x + w]
            try:
                grayFace = cv2.resize(grayFace, (emotionTargetSize))
            except:
                continue

            grayFace = grayFace.astype('float32')
            grayFace = grayFace / 255.0
            grayFace = (grayFace - 0.5) * 2.0
            grayFace = np.expand_dims(grayFace, 0)
            grayFace = np.expand_dims(grayFace, -1)
            emotion_prediction = emotionClassifier.predict(grayFace)
            emotion_probability = np.max(emotion_prediction)
            if (emotion_probability > 0.36):
                emotion_label_arg = np.argmax(emotion_prediction) 
                color = emotions[emotion_label_arg]['color']
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.line(frame, (x, y + h), (x + 20, y + h + 20),
                        color,
                        thickness=2)
                cv2.rectangle(frame, (x + 20, y + h + 20), (x + 110, y + h + 40),
                            color, -1)
                cv2.putText(frame, emotions[emotion_label_arg]['emotion'],
                            (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                emo = emotions[emotion_label_arg]['emotion']
                softmax_emotion = emotion_prediction[0]

            else:
                color = (255, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                emo = 'none'
                none = ['none','none','none','none','none','none','none']
            if 'softmax_emotion' in locals():
                shape_list.append(emo)
                shape_list.extend(softmax_emotion)
                sll.append(shape_list)
            else:
                shape_list.append(emo)
                shape_list.extend(none)
                sll.append(shape_list)
                
        cv2.putText(frame, 'Landmark: '+str(landcount), (0,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        # if args["isVideoWriter"] == True:
        #     videoWrite.write(frame)

        #cv2.imshow("Emotion Recognition", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    coln = ['total_count', 'landmark_count']
    coln += list(range(136))
    coln += ['emotion', emotions[0]['emotion'], emotions[1]['emotion'], emotions[2]['emotion'], 
        emotions[3]['emotion'], emotions[4]['emotion'], emotions[5]['emotion'], emotions[6]['emotion']]
    df = pd.DataFrame(sll, columns=coln)
    df.to_csv('./data/DuckEEs/{}.csv'.format(file_path.stem), index=False)

    cap.release()
    # if args["isVideoWriter"] == True:
    #     videoWrite.release()
    cv2.destroyAllWindows()