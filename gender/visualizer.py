import matplotlib.pyplot as plt
import numpy as np
from utils import get_labels

class Visualizer(object):
    """Visualizer class"""
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.labels = get_labels(self.dataset_name)

    def plot_image(self, image_array, one_hot_encoder=None):
        image_array = np.squeeze(image_array)
        image_array = image_array.astype('uint8')
        plt.imshow(image_array)
        if one_hot_encoder != None:
            label_arg = np.argmax(one_hot_encoder)
            plt.title(self.labels[label_arg])
        plt.show()


