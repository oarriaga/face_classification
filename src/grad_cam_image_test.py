import cv2
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from utils.grad_cam import load_image
from utils.grad_cam import compile_gradient_function
from utils.grad_cam import compile_saliency_function
from utils.grad_cam import register_gradient
from utils.grad_cam import modify_backprop
from utils.grad_cam import calculate_guided_gradient_CAM
from utils.visualizer import display_image
from utils.visualizer import make_mosaic
from utils.visualizer import pretty_imshow
#from utils.grad_cam import reset_optimizer_weights

import pickle
faces = pickle.load(open('utils/faces.pkl','rb'))
# 6 sad, 8 happy, 9 surprise?, 10 angry, 14 happy, 15 surprise?, 22 angry, 25 happy

"""
for face_arg in range(len(faces)):
    print(face_arg)
    display_image(faces[face_arg])
    plt.show()
"""
model_filename = '../trained_models/emotion_models/mini_XCEPTION.523-0.65.hdf5'
#reset_optimizer_weights(model_filename)
model = load_model(model_filename)
grad_cam_faces = []
for face_arg in range(len(faces[:16])):
    face = faces[face_arg]
    preprocessed_input = load_image(face)
    predictions = model.predict(preprocessed_input)
    predicted_class = np.argmax(predictions)
    gradient_function = compile_gradient_function(model, predicted_class, 'conv2d_7')
    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp')
    saliency_function = compile_saliency_function(guided_model)


    guided_gradCAM = calculate_guided_gradient_CAM(preprocessed_input,
                                gradient_function, saliency_function)
    guided_gradCAM = cv2.resize(guided_gradCAM, (128, 128))
    grad_cam_faces.append(guided_gradCAM)
grad_cam_faces = np.asarray(grad_cam_faces)
pretty_imshow(plt.gca(), make_mosaic(grad_cam_faces, 4, 4), cmap='gray')
plt.show()
#cv2.imwrite('guided_gradCAM.jpg', guided_gradCAM)

