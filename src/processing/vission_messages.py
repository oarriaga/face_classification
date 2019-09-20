class Box2D(object):
    """Bounding box 2D coordinates with class label and score.

    # Properties
        coordinates =List of floats/integers containing
            (x_min, y_min, x_max, y_max) coordinates
        class_name: String indicating the class label name of the object.
        center: List of float/integers indicating the center of the box.
        width: Float/integer specifying width of the box.
        height: Float/integer specifying height of the box.
        score: Float. Indicates the score of label associated to the box.
        metadata: Dict. Box metadata, for example image_name
            used for evaluation.

    # Methods
        contains()
    """
    def __init__(self, coordinates, score, class_name=None, metadata={}):
        x_min, y_min, x_max, y_max = coordinates
        if x_min >= x_max:
            raise ValueError('Invalid coordinate input x_min >= x_max')
        if y_min >= y_max:
            raise ValueError('Invalid coordinate input y_min >= y_max')

        self._coordinates = coordinates
        self._class_name = class_name
        self._score = score
        self._metadata = metadata

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        x_min, y_min, x_max, y_max = coordinates
        if x_min >= x_max:
            raise ValueError('Invalid coordinate input x_min >= x_max')
        if y_min >= y_max:
            raise ValueError('Invalid coordinate input y_min >= y_max')

        self._coordinates = coordinates

    @property
    def class_name(self):
        return self._class_name

    @property
    def metadata(self):
        return self._metadata

    @property
    def score(self):
        return self._score

    @property
    def center(self):
        x_center = self._coordinates[0] + self._coordinates[2] / 2.0
        y_center = self._coordinates[1] + self._coordinates[3] / 2.0
        return x_center, y_center

    @property
    def width(self):
        return abs(self.coordinates[2] - self.coordinates[0])

    @property
    def height(self):
        return abs(self.coordinates[3] - self.coordinates[1])

    def __repr__(self):
        return "Box2D({}, {}, {}, {}, {}, {})".format(
            self.coordinates[0], self.coordinates[1],
            self.coordinates[2], self.coordinates[3],
            self.score, self.class_name)

    def contains(self, point):
        """Checks if point is inside bounding box.

        # Arguments
            point: Numpy array of size 2.

        # Returns
            Boolean. 'True' if 'point' is inside bounding box.
                'False' otherwise.
        """
        assert len(point) == 2
        x_min, y_min, x_max, y_max = self.coordinates
        inside_range_x = (point[0] >= x_min) and (point[0] <= x_max)
        inside_range_y = (point[1] >= y_min) and (point[1] <= y_max)
        return (inside_range_x and inside_range_y)
