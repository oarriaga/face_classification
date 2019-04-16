import pandas as pd
import numpy as np
import cv2

IMAGES_PATH = '../datasets/fer2013/fer2013.csv'
LABELS_PATH = '../datasets/fer2013new.csv'


class FERPlus(object):
    """Class for loading FER2013 [1] emotion classification dataset with
    the FERPlus labels [2]:
    [1] kaggle.com/c/challenges-in-representation-learning-facial-\
            expression-recognition-challenge
    [2] github.com/Microsoft/FERPlu://github.com/Microsoft/FERPlus"""

    def __init__(self, split='train', image_size=(48, 48),
                 dataset_name='FERPlus'):

        self.split = split
        self.image_size = image_size
        self.dataset_name = dataset_name
        self.images_path = IMAGES_PATH
        self.labels_path = LABELS_PATH
        self.class_names = ['neutral', 'happiness', 'surprise', 'sadness',
                            'anger', 'disgust', 'fear', 'contempt']
        self.num_classes = len(self.class_names)
        self.arg_to_name = dict(zip(range(self.num_classes), self.class_names))
        self.name_to_arg = dict(zip(self.class_names, range(self.num_classes)))
        self._split_to_filter = {
            'train': 'Training', 'val': 'PublicTest', 'test': 'PrivateTest'}

    def load_data(self):
        filter_name = self._split_to_filter[self.split]
        pixel_sequences = pd.read_csv(self.images_path)
        pixel_sequences = pixel_sequences[pixel_sequences.Usage == filter_name]
        pixel_sequences = pixel_sequences['pixels'].tolist()
        faces = []
        for pixel_sequence in pixel_sequences:
            face = [float(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(48, 48)
            faces.append(cv2.resize(face, self.image_size))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)

        emotions = pd.read_csv(self.labels_path)
        emotions = emotions[emotions.Usage == filter_name]
        emotions = emotions.iloc[:, 2:10].values
        N = np.sum(emotions, axis=1)
        mask = N != 0
        N, faces, emotions = N[mask], faces[mask], emotions[mask]
        emotions = emotions / np.expand_dims(N, 1)
        return faces, emotions
