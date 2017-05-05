"""
File: train_emotion_classifier.py
Author: Octavio Arriaga
Email: arriaga.camargo@gmail.com
Github: https://github.com/oarriaga
Description: Train emotion classification model
"""
from keras.callbacks import CSVLogger, ModelCheckpoint
from data_loader import DataLoader
from models import simple_CNN
from utils import preprocess_input

# parameters
batch_size = 32
num_epochs = 100
training_split = .8
dataset_name = 'fer2013'
log_file_path = 'log_files/emotion_training.log'
trained_models_path = '../trained_models/emotion_models/simple_CNN'

# data loader
data_loader = DataLoader(dataset_name)
faces, emotions = data_loader.get_data()
faces = preprocess_input(faces)
num_classes = emotions.shape[1]
input_shape = faces.shape[1:]

# model parameters/compilation
model = simple_CNN(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
                                        metrics=['accuracy'])
model.summary()

# model callbacks
csv_logger = CSVLogger(log_file_path, append=False)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names,
                            'val_acc', verbose=1,
                                save_best_only=True)

# model training
model.fit(faces, emotions, batch_size, num_epochs,verbose=1,
                    callbacks=[model_checkpoint, csv_logger],
                        validation_split=(1-training_split),
                                                shuffle=True)
