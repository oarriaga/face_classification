import sys

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
from utils.inference import draw_bounding_box
from utils.inference import load_image


# parameters
image_path = sys.argv[1]
# task = sys.argv[2]
task = 'emotion'
if task == 'emotion':
    labels = get_labels('fer2013')
    offsets = (0, 0)
    # model_filename = '../trained_models/fer2013_big_XCEPTION.54-0.66.hdf5'
    model_filename = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
elif task == 'gender':
    labels = get_labels('imdb')
    offsets = (30, 60)
    model_filename = '../trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5'

color = (0, 255, 0)

# loading models
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
model = load_model(model_filename, compile=False)
target_size = model.input_shape[1:3]
face_detection = load_detection_model(detection_model_path)

# loading images
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')
faces = detect_faces(face_detection, gray_image)

# start prediction for every image
for face_coordinates in faces:

    x1, x2, y1, y2 = apply_offsets(face_coordinates, offsets)
    rgb_face = rgb_image[y1:y2, x1:x2]

    x1, x2, y1, y2 = apply_offsets(face_coordinates, offsets)
    gray_face = gray_image[y1:y2, x1:x2]

    # processing input
    try:
        gray_face = cv2.resize(gray_face, (target_size))
    except:
        continue
    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)

    # prediction
    predicted_class = np.argmax(model.predict(gray_face))
    label_text = labels[predicted_class]

    gradient_function = compile_gradient_function(model,
                            predicted_class, 'conv2d_7')
    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp', task)
    saliency_function = compile_saliency_function(guided_model, 'conv2d_7')

    guided_gradCAM = calculate_guided_gradient_CAM(gray_face,
                        gradient_function, saliency_function)
    guided_gradCAM = cv2.resize(guided_gradCAM, (x2-x1, y2-y1))
    rgb_guided_gradCAM = np.repeat(guided_gradCAM[:, :, np.newaxis], 3, axis=2)
    rgb_image[y1:y2, x1:x2, :] = rgb_guided_gradCAM
    draw_bounding_box((x1, y1, x2 - x1, y2 - y1), rgb_image, color)
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('../images/guided_gradCAM.png', bgr_image)
