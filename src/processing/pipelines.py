import cv2
import numpy as np
from .vission_messages import Box2D
from .utils import lincolor


class DetectionPipeline(object):
    def __init__(self, detector, classifier, offsets, class_names):
        self.detector = detector
        self.classifier = classifier
        self.offsets = offsets
        self.class_names = class_names
        self.colors = lincolor(len(class_names))

    def __call__(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.predict(gray_image)
        boxes2D = []
        for coordinates in faces:
            coordinates = self.apply_offsets(coordinates, self.offsets)
            x_min, x_max, y_min, y_max = coordinates
            face = gray_image[y_min:y_max, x_min:x_max]
            if ((face.shape[0] == 0) or (face.shape[1] == 0)):
                continue
            face = cv2.resize(face, self.classifier.input_shape[1:3])
            face = np.expand_dims(np.expand_dims(face, -1), 0) / 255.0
            emotions = self.classifier.predict(face)
            probability, class_arg = np.max(emotions), np.argmax(emotions)
            class_name = self.class_names[class_arg]
            color = probability * np.asarray(self.colors[class_arg])

            self.draw_bounding_box(coordinates, image, color)
            text_coordinates = (coordinates[0] + 0, coordinates[2] - 10)
            self.draw_text(text_coordinates, image, class_name, color)

            coordinates = (x_min, y_min, x_max, y_max)
            box2D = Box2D(coordinates, probability, class_name)
            boxes2D.append(box2D)
        return {'boxes2D': boxes2D, 'image': image}

    def apply_offsets(self, coordinates, offsets):
        x, y, width, height = coordinates
        x_off, y_off = offsets
        return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

    def draw_bounding_box(self, coordinates, image, color):
        x_min, x_max, y_min, y_max = coordinates
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    def draw_text(self, coordinates, image, text, color, scale=1, thick=1):
        font, line = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
        cv2.putText(image, text, coordinates, font, scale, color, thick, line)
