import sys

import cv2
import numpy as np
from keras.models import load_model
from utils.grad_cam import compile_gradient_function
from utils.grad_cam import compile_saliency_function
from utils.grad_cam import register_gradient
from utils.grad_cam import modify_backprop
from utils.grad_cam import calculate_guided_gradient_CAM
from utils.inference import detect_faces
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from utils.inference import draw_bounding_box
from utils.datasets import get_class_to_arg

# getting the correct model given the input
# task = sys.argv[1]
# class_name = sys.argv[2]
task = 'emotion'
if task == 'gender':
    model_filename = '../trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5'
    class_to_arg = get_class_to_arg('imdb')
    # predicted_class = class_to_arg[class_name]
    predicted_class = 0
    offsets = (0, 0)
elif task == 'emotion':
    model_filename = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    # model_filename = '../trained_models/fer2013_big_XCEPTION.54-0.66.hdf5'
    class_to_arg = get_class_to_arg('fer2013')
    # predicted_class = class_to_arg[class_name]
    predicted_class = 0
    offsets = (0, 0)

model = load_model(model_filename, compile=False)
gradient_function = compile_gradient_function(model, predicted_class, 'conv2d_7')
register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp', task)
saliency_function = compile_saliency_function(guided_model, 'conv2d_7')

# parameters for loading data and images 
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
face_detection = load_detection_model(detection_model_path)
color = (0, 255, 0)

# getting input model shapes for inference
target_size = model.input_shape[1:3]

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

        x1, x2, y1, y2 = apply_offsets(face_coordinates, offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        guided_gradCAM = calculate_guided_gradient_CAM(gray_face,
                            gradient_function, saliency_function)
        guided_gradCAM = cv2.resize(guided_gradCAM, (x2-x1, y2-y1))
        try:
            rgb_guided_gradCAM = np.repeat(guided_gradCAM[:, :, np.newaxis],
                                                                3, axis=2)
            rgb_image[y1:y2, x1:x2, :] = rgb_guided_gradCAM
        except:
            continue
        draw_bounding_box((x1, y1, x2 - x1, y2 - y1), rgb_image, color)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    try:
        cv2.imshow('window_frame', bgr_image)
    except:
        continue
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


