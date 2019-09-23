from keras.utils import to_categorical
import numpy as np
import cv2

from .utils import get_labels


FER_PATH = '../datasets/fer2013/fer2013.csv'


class FER(object):
    """Class for loading FER2013 [1] emotion classification dataset.
    [1] kaggle.com/c/challenges-in-representation-learning-facial-\
            expression-recognition-challenge
    """

    def __init__(self, split='train', image_size=(48, 48), dataset_name='FER'):
        self.split = split
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.path = FER_PATH
        self.class_names = get_labels(self.dataset_name)
        self.num_classes = len(self.class_names)
        self._split_to_filter = {
            'train': 'Training', 'val': 'PublicTest', 'test': 'PrivateTest'}

    def load_data(self):
        data = np.genfromtxt(self.path, str, delimiter=',', skip_header=1)
        data = data[data[:, -1] == self._split_to_filter[self.split]]
        faces = np.zeros((len(data), *self.image_size, 1))
        for sample_arg, sample in enumerate(data):
            face = np.array(sample[1].split(' '), dtype=int).reshape(48, 48)
            faces[sample_arg, :, :, 0] = cv2.resize(face, self.image_size)
        emotions = to_categorical(data[:, 0].astype(int), self.num_classes)
        return faces, emotions
