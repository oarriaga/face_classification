import numpy as np
import cv2

from .utils import get_labels

IMAGES_PATH = '../datasets/fer2013/fer2013.csv'
LABELS_PATH = '../datasets/fer2013/fer2013new.csv'


class FERPlus(object):
    """Class for loading FERPlus. FERPlus contains faces from FER [1] with
    the new labels made in [2]:
    [1] kaggle.com/c/challenges-in-representation-learning-facial-\
            expression-recognition-challenge
    [2] https://github.com/microsoft/FERPlus/blob/master/fer2013new.csv"""

    def __init__(self, split='train', image_size=(48, 48),
                 dataset_name='FERPlus'):

        self.split = split
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.images_path = IMAGES_PATH
        self.labels_path = LABELS_PATH
        self.class_names = get_labels('FERPlus')
        self.num_classes = len(self.class_names)
        self.split_to_filter = {
            'train': 'Training', 'val': 'PublicTest', 'test': 'PrivateTest'}

    def load_data(self):
        data = np.genfromtxt(self.images_path, str, '#', ',', 1)
        data = data[data[:, -1] == self.split_to_filter[self.split]]
        faces = np.zeros((len(data), *self.image_size, 1))
        for sample_arg, sample in enumerate(data):
            face = np.array(sample[1].split(' '), dtype=int).reshape(48, 48)
            faces[sample_arg, :, :, 0] = cv2.resize(face, self.image_size)

        emotions = np.genfromtxt(self.labels_path, str, '#', ',', 1)
        emotions = emotions[emotions[:, 0] == self.split_to_filter[self.split]]
        emotions = emotions[:, 2:10].astype(float)
        N = np.sum(emotions, axis=1)
        mask = N != 0
        N, faces, emotions = N[mask], faces[mask], emotions[mask]
        emotions = emotions / np.expand_dims(N, 1)
        return faces, emotions
