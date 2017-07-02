import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from util.preprocessor import preprocess_input

# parameters for loading data and images 
image_path = '../images/test_image.jpg'
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/simple_CNN.530-0.65.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
emotion_offsets = (20, 40)
gender_offsets = (30, 60)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path)
gender_classifier = load_model(gender_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:]
gender_target_size = gender_classifier.input_shape[1:]

# loading images
gray_image = image.load_img(image_path, grayscale=True)
rgb_image = image.load_img(image_path, grayscale=False)

faces = detect_faces(face_detection, gray_image)
for face_coordinates in faces:

    x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
    rgb_face = rgb_image[y1:y2, x1:x2]

    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    gray_face = gray_image[y1:y2, x1:x2]

    rgb_face = cv2.resize(rgb_face, gender_target_size)
    gray_face = cv2.resize(gray_face, emotion_target_size)

    rgb_face = np.expand_dims(rgb_face, 0)
    egb_face = preprocess_input(rgb_face)
    gender_prediction = gender_classifier.predict(rgb_face)
    gender_label_arg = np.argmax(gender_prediction)
    gender = gender_labels[gender_label_arg]

    gray_face = preprocess_input(gray_face)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion = emotion_labels[emotion_label_arg]

    if gender == gender_labels[0]:
        gender_color = (0, 0, 255)
    else:
        gender_color = (255, 0, 0)

    cv2.rectangle(frame, (x, y), (x + w, y + h), gender_color, 2)
    cv2.putText(frame, emotion, (x, y - 90), font,
                    2, gender_color, 2, cv2.LINE_AA)
    cv2.putText(frame, gender, (x , y - 90 + 70), font,
                    2, gender_color, 2, cv2.LINE_AA)

frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imwrite('../images/predicted_test_image.png', frame)


