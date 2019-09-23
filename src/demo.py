from processing import DetectionPipeline
from processing import VideoPlayer
from models import HaarCascadeDetector
from keras.models import load_model
from datasets import get_labels


# parameters for loading data and images
detector_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
classifier_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
image_size, offsets, class_names = (640, 480), (20, 40), get_labels('FER')

# loading models
classifier = load_model(classifier_path, compile=False)
detector = HaarCascadeDetector(detector_path)
pipeline = DetectionPipeline(detector, classifier, offsets, class_names)
video_player = VideoPlayer(image_size, pipeline, camera=0)
video_player.start()

# check if adding batch normalization is done in original Xception
# model = build_xception((64, 64, 1), 7, [8, 8], [16, 32, 64, 128])
