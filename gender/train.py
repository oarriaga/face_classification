from keras.callbacks import CSVLogger, ModelCheckpoint
from data_loader import DataLoader
from models import attention_CNN
from image_generator import ImageGenerator
from utils import split_data

batch_size = 32
num_classes = 2
num_epochs = 10
input_shape = (50, 50, 3)
trained_models_path = '../trained_models/attention_weights'

data_loader = DataLoader('imdb')
ground_truth_data = data_loader.load_dataset()
train_keys, val_keys = split_data(ground_truth_data, training_ratio=.8)
image_generator = ImageGenerator(ground_truth_data, batch_size, input_shape[:2],
                                train_keys, val_keys, None,
                                path_prefix='../datasets/imdb_crop/',
                                do_crop=True)

model = attention_CNN(input_shape, num_classes)
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model_names = trained_models_path + '.{epoch:02d}-{val_loss:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False)

csv_logger = CSVLogger('training.log')
model.fit_generator(image_generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_keys)/batch_size),
                    epochs=num_epochs, verbose=1,
                    callbacks=[csv_logger],
                    validation_data= image_generator.flow('val'),
                    validation_steps=int(len(val_keys)/batch_size))
