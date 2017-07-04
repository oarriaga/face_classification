import cv2
import numpy as np
from keras.models import load_model
from utils.grad_cam import compile_gradient_function
from utils.grad_cam import compile_saliency_function
from utils.grad_cam import register_gradient
from utils.grad_cam import modify_backprop
from utils.grad_cam import calculate_guided_gradient_CAM
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

model_filename = '../trained_models/emotion_models/mini_XCEPTION.523-0.65.hdf5'
model = load_model(model_filename)
predicted_class = 3
gradient_function = compile_gradient_function(model, predicted_class, 'conv2d_7')
register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp')
saliency_function = compile_saliency_function(guided_model)

# parameters for loading data and images 
#image_path = '../images/test_image.jpg'
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/mini_XCEPTION.523-0.65.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        guided_gradCAM = calculate_guided_gradient_CAM(gray_face,
                            gradient_function, saliency_function)
        guided_gradCAM = cv2.resize(guided_gradCAM, (x2-x1, y2-y1))
        try:
            gray_image[y1:y2, x1:x2] = guided_gradCAM
        except:
            continue
        """
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))
        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                                        color, 0, -45, 1, 1)
        """
    #bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    #cv2.imshow('window_frame', bgr_image)
    try:
        cv2.imshow('window_frame', gray_image)
    except:
        continue
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


