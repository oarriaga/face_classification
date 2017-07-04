import pickle

from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

from utils.visualizer import pretty_imshow
from utils.visualizer import make_mosaic
from utils.preprocessor import preprocess_input

model = load_model('../trained_models/gender_models/simple_CNN.81-0.96.hdf5')
conv_weights = model.layers[1].get_weights()
kernel_conv_weights = conv_weights[0]
kernel_conv_weights = np.squeeze(kernel_conv_weights)
kernel_conv_weights = np.rollaxis(kernel_conv_weights, 2, 0)
kernel_conv_weights = np.expand_dims(kernel_conv_weights, -1)
num_kernels = kernel_conv_weights.shape[0]
box_size = int(np.ceil(np.sqrt(num_kernels)))
print('Box size:', box_size)

print('Kernel shape', kernel_conv_weights.shape)
plt.figure(figsize=(15, 15))
plt.title('conv1 weights')
pretty_imshow(plt.gca(),
        make_mosaic(kernel_conv_weights, box_size, box_size),
        cmap='gray')
plt.show()

get_feature_map = K.function([model.layers[0].input, K.learning_phase()],
                                                [model.layers[1].output])

faces = pickle.load(open('utils/faces.pkl', 'rb'))
face = faces[0]
face = preprocess_input(face)
face = np.expand_dims(face, 0)
feature_map = get_feature_map((face,False))[0]
feature_map = np.rollaxis(feature_map, 3, 0)
feature_map = np.rollaxis(feature_map, 1, 4)
print(feature_map.shape)
pretty_imshow(plt.gca(),
        make_mosaic(feature_map, box_size, box_size),
        cmap='gray')
plt.show()


pretty_imshow(plt.gca(),
        make_mosaic(faces[:16], 4, 4),
        cmap='gray')
plt.show()

