import numpy as np
from scipy.io import loadmat


class IMDB(object):
    """Class for loading the IMDB face sex classification dataset"""
    def __init__(self, dataset_path, split='train', image_size=(48, 48),
                 dataset_name='IMDB'):

        self.dataset_path = dataset_path
        self.image_size = image_size
        self.dataset_name = dataset_name
        if self.dataset_path is not None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'imdb':
            self.dataset_path = '../datasets/imdb_crop/imdb.mat'
        else:
            raise Exception(
                'Incorrect dataset name, please input imdb or fer2013')

    def load_data(self):
        face_score_treshold = 3
        dataset = loadmat(self.dataset_path)
        image_names_array = dataset['imdb']['full_path'][0, 0][0]
        sex_classes = dataset['imdb']['gender'][1, 0][0]
        face_score = dataset['imdb']['face_score'][0, 0][0]
        second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
        face_score_mask = face_score > face_score_treshold
        second_face_score_mask = np.isnan(second_face_score)
        unknown_gender_mask = np.logical_not(np.isnan(sex_classes))
        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        image_names_array = image_names_array[mask]
        sex_classes = sex_classes[mask].tolist()
        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)
        return image_names, sex_classes
