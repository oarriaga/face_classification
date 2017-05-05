"""
File: train_gender_classifier.py
Author: Octavio Arriaga
Email: arriaga.camargo@gmail.com
Github: https://github.com/oarriaga
Description: Train gender classification model
"""

from keras.callbacks import CSVLogger, ModelCheckpoint
from data_loader import DataLoader
from models import simple_CNN, attention_CNN
from image_generator import ImageGenerator
from utils import split_data

# Parameters and given variables
batch_size = 32
num_epochs = 30
num_classes = 2
training_ratio = .8
dataset_name = 'imdb'
do_random_crop = True
input_shape = (48, 48, 3)
images_path = '../datasets/imdb_crop/'
log_file_path = 'log_files/gender_training.log'
trained_models_path = '../trained_models/gender_models/simple_CNN'

data_loader = DataLoader(dataset_name)
ground_truth_data = data_loader.get_data()
train_keys, val_keys = split_data(ground_truth_data, training_ratio)
image_generator = ImageGenerator(ground_truth_data, batch_size, input_shape[:2],
                                train_keys, val_keys, None,
                                path_prefix=images_path,
                                vertical_flip_probability=0,
                                do_random_crop=do_random_crop)

model = attention_CNN(input_shape, num_classes)
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
model.summary()

model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False)

csv_logger = CSVLogger(log_file_path, append=False)
model.fit_generator(image_generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_keys) / batch_size),
                    epochs=num_epochs, verbose=1,
                    callbacks=[csv_logger, model_checkpoint],
                    validation_data= image_generator.flow('val'),
                    validation_steps=int(len(val_keys) / batch_size))

