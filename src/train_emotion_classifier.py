"""
File: train_emotion_classifier.py
Author: Octavio Arriaga
Email: arriaga.camargo@gmail.com
Github: https://github.com/oarriaga
Description: Train emotion classification model
"""

from keras.callbacks import CSVLogger, ModelCheckpoint
from utils.data_manager import DataManager
from models.cnn import mini_XCEPTION
from utils.utils import preprocess_input
from utils.utils import split_data
from keras.preprocessing.image import ImageDataGenerator

# parameters
batch_size = 128
num_epochs = 1000
validation_split = .2
verbose = 1
dataset_name = 'fer2013'
log_file_path = '../trained_models/emotion_models/emotion_training.log'
trained_models_path = '../trained_models/emotion_models/mini_XCEPTION'

# data loader
data_loader = DataManager(dataset_name)
faces, emotions = data_loader.get_data()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
input_shape = faces.shape[1:]

# model parameters/compilation
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
                                        metrics=['accuracy'])
model.summary()

# model callbacks
csv_logger = CSVLogger(log_file_path, append=False)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names,
                            'val_acc', verbose=1,
                                save_best_only=True)
callbacks = [model_checkpoint, csv_logger]


# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=20,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

train_data, val_data = split_data(faces, emotions, validation_split)
train_faces, train_emotions = train_data

model.fit_generator(data_generator.flow(train_faces, train_emotions, batch_size),
                    steps_per_epoch=len(train_faces) / batch_size,
                    epochs=num_epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_data)


