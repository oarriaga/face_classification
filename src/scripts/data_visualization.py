from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

from datasets import FERPlus
from utils import make_mosaic, normal_imshow


random_state = 777

data_manager = FERPlus(split='train', image_size=(128, 128))
print('Loading dataset...')
images, vector_labels = data_manager.load_data()
num_samples, num_classes = vector_labels.shape

print('Preprocessing dataset...')
labels = np.argmax(vector_labels, axis=1)
frequencies = [np.sum(labels == class_arg) for class_arg in range(num_classes)]
arg_to_name = data_manager.arg_to_name
label_names = [arg_to_name[class_arg] for class_arg in range(num_classes)]
images = images / 255.

print('Plotting frequencies...')
indices = np.arange(num_classes)
plt.bar(indices, frequencies)
plt.xlabel('Class indices')
plt.ylabel('Number of images')
plt.xticks(indices, label_names, rotation=30)
plt.title('FERPlus class distribution')
plt.show()

print('Displaying augmented samples')
generator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True)

generator.fit(images)
generated_images = next(generator.flow(images))
normal_imshow(plt.gca(), make_mosaic(generated_images[:9], 3, 3))
plt.show()
