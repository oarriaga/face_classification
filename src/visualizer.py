import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from keras.models import load_model
#import matplotlib.cm as cm
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import pylab as pl

dataset_name = 'fer2013'
model_path = '../trained_models/emotion_models/simple_CNN.985-0.66.hdf5'

model = load_model(model_path)
conv_2_function = K.function([model.input], [model.layers[2].output])

#def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
#    if cmap is None:
#        cmap = cm.jet
#    if vmin is None:
#        vmin = data.min()
#    if vmax is None:
#        vmax = data.max()
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
#    pl.colorbar(im, cax=cax)
#

def display_image(face, class_vector=None, class_decoder=None):
    if class_vector is not None and class_decoder is None:
        raise Exception('Provide class decoder')
    face = np.squeeze(face)
    color_map = None
    if len(face.shape) < 3:
        color_map = 'gray'
    plt.imshow(face, color_map)
    if class_vector is not None:
        class_arg = np.argmax(class_vector)
        class_name = class_decoder[class_arg]
        plt.title(class_name)
    plt.show()

"""
if __name__ == '__main__':
    from utils.data_manager import DataManager
    from utils.utils import get_labels

    dataset_name = 'fer2013'
    model_path = '../trained_models/emotion_models/simple_CNN.985-0.66.hdf5'
    class_decoder = get_labels(dataset_name)
    data_manager = DataManager(dataset_name)
    faces, emotions = data_manager.get_data()
    image_arg = 0
    face = faces[image_arg:image_arg + 1]
    emotion = emotions[image_arg:image_arg + 1]
    display_image(face, emotion, class_decoder)
"""
