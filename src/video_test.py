import cv2
from keras.models import load_model
import numpy as np
from statistics import mode
from utils import preprocess_input
from utils import get_labels

detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
gender_model_path = '../trained_models/gender_models/simple_CNN.87-0.91.hdf5'
frame_window = 10
x_offset = 30
y_offset = 60
gender_labels = get_labels('imdb')

face_detection = cv2.CascadeClassifier(detection_model_path)
gender_classifier = load_model(gender_model_path)
label_window = []
video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.startWindowThread()
cv2.namedWindow('window_frame')
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detection.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x - x_offset, y - y_offset),
                    (x + w + x_offset, y + h + y_offset),
                    (255, 0, 0), 2)
        face = frame[(y - y_offset):(y + h + y_offset),
                    (x - x_offset):(x + w + x_offset)]
        try:
            face = cv2.resize(face, (48, 48))
        except:
            continue
        face = np.expand_dims(face, 0)
        face = preprocess_input(face)
        label_arg = np.argmax(gender_classifier.predict(face))
        gender = gender_labels[label_arg]
        label_window.append(gender)

        if len(label_window) >= frame_window:
            label_window.pop(0)
        try:
            gender_mode = mode(label_window)
        except:
            continue
        cv2.putText(frame, gender_mode, (x, y - 30), font,
                        .7, (255, 0, 0), 1, cv2.LINE_AA)

    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', frame)
    except:
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

