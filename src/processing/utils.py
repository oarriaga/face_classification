import colorsys
import random
import cv2


def load_image(image_path, grayscale=False, size=None):
    """ loads image as array
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(image_path, flag)
    if size is not None:
        image = cv2.resize(image, size)
    return image


def lincolor(num_colors, saturation=1, value=1, normalized=False):
    """Creates a list of RGB colors linearly sampled from HSV space with
    randomised Saturation and Value

    # Arguments
        num_colors: Integer.
        saturation: Float or `None`. If float indicates saturation.
            If `None` it samples a random value.
        value: Float or `None`. If float indicates value.
            If `None` it samples a random value.

    # Returns
        List, for which each element contains a list with RGB color

    # References
        [Original implementation](https://github.com/jutanke/cselect)
    """
    RGB_colors = []
    hues = [value / num_colors for value in range(0, num_colors)]
    for hue in hues:

        if saturation is None:
            saturation = random.uniform(0.6, 1)

        if value is None:
            value = random.uniform(0.5, 1)

        RGB_color = colorsys.hsv_to_rgb(hue, saturation, value)
        if not normalized:
            RGB_color = [int(color * 255) for color in RGB_color]
        RGB_colors.append(RGB_color)
    return RGB_colors


class VideoPlayer(object):
    """Performs and visualizes inferences in a real-time video.

    # Properties
        image_size: List of two integers. Output size of the displayed image.
        pipeline: Function. Should take image as input and it should output a
            dictionary containing as keys ``image`` and ``boxes2D``.

    # Methods
        start()
    """

    def __init__(self, image_size, pipeline, camera=0):
        self.image_size = image_size
        self.pipeline = pipeline
        self.camera = camera

    def start(self):
        camera = cv2.VideoCapture(self.camera)
        while True:
            frame = camera.read()[1]
            if frame is None:
                print('Frame: None')
                continue

            results = self.pipeline(image=frame)
            image = cv2.resize(results['image'], tuple(self.image_size))
            cv2.imshow('webcam', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        camera.release()
        cv2.destroyAllWindows()
