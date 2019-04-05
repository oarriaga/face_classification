import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse

from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Nadam

from datasets import FERPlus
from datasets import SAVE_PATH
from models import build_xception


description = 'Training real-time CNNs for emotion and sex classification'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--dataset', default='FERPlus',
                    choices=['FERPlus', 'IMDB'], type=str,
                    help='FERPlus or IMDB')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--learning_rate', default=0.002, type=float,
                    help='Initial learning rate')
parser.add_argument('--image_size', default=128, type=int,
                    help='Size of the side of a square image e.g. 64')
parser.add_argument('--stop_patience', default=12, type=int,
                    help='Number of epochs before doing early stopping')
parser.add_argument('--plateau_patience', default=5, type=int,
                    help='Number of epochs before reducing learning rate')
args = parser.parse_args()

if args.dataset == 'FERPlus':
    num_classes = 8
    class_weight = [1.0, 1.3, 2.8, 2.9, 4.1, 53.5, 15.8, 62.2]
elif args.dataset == 'IMDB':
    num_classes = 1

input_shape = (128, 128, 1)
kernels_per_module = [128, 128, 256, 256, 512, 512, 1024]
stop_patience = 12
loss = 'categorical_crossentropy'
metrics = ['accuracy']
splits = ['train', 'val']
max_num_epochs = 10000

# loading datasets
datasets = []
for split in splits:
    data_manager = FERPlus(split, input_shape[:2])
    datasets.append(data_manager.load_data())

# data generator
train_data = (datasets[0][0] / 255., datasets[0][1])
val_data = (datasets[1][0] / 255., datasets[1][1])

data_augmentator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True)

# model
optimizer = Nadam(args.learning_rate)
model = build_xception(input_shape, num_classes, kernels_per_module)
model.compile(optimizer, loss, metrics)
model.summary()

# callbacks
model_path = os.path.join(SAVE_PATH, model.name)
if not os.path.exists(model_path):
    os.makedirs(model_path)
logger = CSVLogger(os.path.join(model_path, model.name + '-optimization.log'))
save_path = os.path.join(model_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(save_path, verbose=1, save_weights_only=True)
early_stop = EarlyStopping(patience=stop_patience)
reduce_lr = ReduceLROnPlateau(patience=args.plateau_patience, verbose=1)
callbacks = [checkpoint, logger, early_stop, reduce_lr]

# training
model.fit_generator(
    data_augmentator.flow(train_data[0], train_data[1], args.batch_size),
    steps_per_epoch=len(train_data[0]) / args.batch_size,
    epochs=max_num_epochs,
    callbacks=callbacks,
    class_weight=class_weight,
    validation_data=val_data,
    use_multiprocessing=True,
    workers=4)
