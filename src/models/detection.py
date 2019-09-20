import cv2


class HaarCascadeDetector(object):
    def __init__(self, path, scale_factor=1.3, min_neighbors=5):
        self.model = cv2.CascadeClassifier(path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def predict(self, gray_image):
        """ detects faces
        """
        if len(gray_image.shape) != 2:
            raise ValueError('Invalid gray image shape:', gray_image.shape)
        args = (gray_image, self.scale_factor, self.min_neighbors)
        return self.model.detectMultiScale(*args)
