from data_loader import DataLoader
from image_generator import ImageGenerator
from visualizer import Visualizer
from utils import split_data

batch_size = 100
num_classes = 2
input_shape = (50, 50, 3)

data_loader = DataLoader('imdb')
ground_truth_data = data_loader.load_dataset()
train_keys, val_keys = split_data(ground_truth_data, training_ratio=.8)
image_generator = ImageGenerator(ground_truth_data, batch_size, input_shape[:2],
                                train_keys, val_keys, None,
                                path_prefix='../datasets/imdb_crop/',
                                vertical_flip_probability=0,
                                do_crop=False)
visualizer = Visualizer('imdb')
for image_arg in range(batch_size):
    image_batch = next(image_generator.flow('demo'))
    image_array = image_batch[0]['image_array_input'][image_arg]
    one_hot_encoding = image_batch[1]['predictions'][image_arg]
    visualizer.plot_image(image_array, one_hot_encoding)
