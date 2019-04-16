import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from datetime import datetime
import json

from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator

from datasets import FERPlus
from models import build_xception, build_densenet, build_vgg


description = 'Training real-time CNNs for emotion and sex classification'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--dataset', default='FERPlus',
                    choices=['FERPlus', 'IMDB'], type=str,
                    help='FERPlus or IMDB')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--learning_rate', default=0.002, type=float,
                    help='Initial learning rate for Nadam')
parser.add_argument('--image_size', default=128, type=int,
                    help='Size of the side of a square image e.g. 64')
parser.add_argument('--stop_patience', default=9, type=int,
                    help='Number of epochs before doing early stopping')
parser.add_argument('--plateau_patience', default=4, type=int,
                    help='Number of epochs before reducing learning rate')
parser.add_argument('--max_num_epochs', default=10000, type=int,
                    help='Maximum number of epochs before finishing')
parser.add_argument('--loss', default='categorical_crossentropy', type=str,
                    help='Classification loss')
parser.add_argument('--stem_kernels', nargs='+', default=[32, 64], type=int,
                    help='Kernels used in the stem block e.g. 32, 64')
parser.add_argument('--block_data', nargs='+',
                    default=[128, 128, 256, 256, 512, 512, 1024], type=int,
                    help='If model is `vgg` or `xception` list should \
                    contain the  number of kernels per block. If `densenet` \
                    list contains the number of dense blocks per \
                    transition block')
parser.add_argument('--class_weight', nargs='+', type=int,
                    default=[1.0, 1.3, 2.8, 2.9, 4.1, 53.5, 15.8, 62.2],
                    help='Kernels used in each block e.g. 128 256 512')
parser.add_argument('--model_name', default='xception',
                    choices=['xception', 'densenet', 'vgg'], type=str,
                    help='CNN model structure')
parser.add_argument('--save_path', default='../trained_models/', type=str,
                    help='Path for writing model weights and logs')
args = parser.parse_args()

if args.dataset == 'FERPlus':
    DataManager = FERPlus
else:
    raise NotImplementedError('IMDB dataset not implemented')

# loading datasets
data_managers, datasets = [], []
input_shape = (args.image_size, args.image_size, 1)
for split in ['train', 'val', 'test']:
    data_manager = DataManager(split, input_shape[:2])
    data_managers.append(data_manager)
    faces, labels = data_managers.load_data()
    datasets.append([faces / 255.0, labels])

# data generator and augmentations
data_augmentator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True)

# instantiating model
model_builder = {
    'xception': build_xception, 'densenet': build_densenet, 'vgg': build_vgg}
optimizer = Nadam(args.learning_rate)
num_classes = data_manager[0].num_classes
model_builder = model_builder[args.model_name]
model_inputs = (input_shape, num_classes, args.stem_kernels, args.block_data)
model = model_builder(*model_inputs)
model.compile(optimizer, args.loss, metrics=['accuracy'])
model.summary()

# setting callbacks and saving hyper-parameters
date = datetime.now().strftime('_%d-%m-%Y_%H:%M:%S')
save_path = os.path.join(args.save_path, args.dataset, model.name + date)
if not os.path.exists(save_path):
    os.makedirs(save_path)
logger = CSVLogger(os.path.join(save_path, model.name + '_optimization.log'))
weights_name = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
weights_path = os.path.join(save_path, weights_name)
checkpoint = ModelCheckpoint(weights_path, verbose=1, save_weights_only=True)
early_stop = EarlyStopping(patience=args.stop_patience)
reduce_lr = ReduceLROnPlateau(patience=args.plateau_patience, verbose=1)
callbacks = [checkpoint, logger, early_stop, reduce_lr]
with open(os.path.join(save_path, 'hyperparameters.json'), 'w') as filer:
    json.dump(args.__dict__, filer, indent=4)
with open(os.path.join(save_path, 'model_summary.txt'), 'w') as filer:
    model.summary(print_fn=lambda x: filer.write(x + '\n'))

# training model
train_data, val_data, test_data = datasets
model.fit_generator(
    data_augmentator.flow(train_data[0], train_data[1], args.batch_size),
    steps_per_epoch=len(train_data[0]) / args.batch_size,
    epochs=args.max_num_epochs,
    callbacks=callbacks,
    class_weight=args.class_weight,
    validation_data=val_data,
    use_multiprocessing=True,
    workers=4)

# writing evaluations
evaluations = model.evaluate(test_data[0], test_data[1])
with open(os.path.join(save_path, 'evaluation.txt'), 'w') as filer:
    for evaluation in evaluations:
        filer.write("%s\n" % evaluation)
print('evaluations:', evaluations)
