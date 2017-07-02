from scipy.io import loadmat
import pandas as pd
import numpy as np
from random import shuffle

class DataManager(object):
    """Class for loading fer2013 emotion classification dataset or
        imdb gender classification dataset."""
    def __init__(self, dataset_name='imdb', dataset_path=None):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'imdb':
            self.dataset_path = '../datasets/imdb_crop/imdb.mat'
        elif self.dataset_name == 'fer2013':
            self.dataset_path = '../datasets/fer2013/fer2013.csv'
        else:
            raise Exception('Incorrect dataset name, please input imdb or fer2013')

    def get_data(self):
        if self.dataset_name == 'imdb':
            ground_truth_data = self._load_imdb()
        elif self.dataset_name == 'fer2013':
            ground_truth_data = self._load_fer2013()
        return ground_truth_data

    def _load_imdb(self):
        face_score_treshold = 3
        dataset = loadmat(self.dataset_path)
        image_names_array = dataset['imdb']['full_path'][0, 0][0]
        gender_classes = dataset['imdb']['gender'][0, 0][0]
        face_score = dataset['imdb']['face_score'][0, 0][0]
        second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
        face_score_mask = face_score > face_score_treshold
        second_face_score_mask = np.isnan(second_face_score)
        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        image_names_array = image_names_array[mask]
        gender_classes = gender_classes[mask].tolist()
        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)
        return dict(zip(image_names, gender_classes))

    def _load_fer2013(self):
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            faces.append(face)
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions

def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}
    elif dataset_name == 'imdb':
        return {0:'woman', 1:'man'}
    else:
        raise Exception('Invalid dataset name')

def split_imdb_data(ground_truth_data, training_ratio=.8, do_shuffle=False):
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle == True:
        shuffle(ground_truth_keys)
    num_train = int(round(training_ratio * len(ground_truth_keys)))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys

def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

