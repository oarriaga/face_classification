from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.optimizers import SGD
from models import simple_CNN
from utils import load_data, preprocess_input
import keras.backend as K
import tensorflow as tf

data_path = '../datasets/fer2013/fer2013.csv'
model_save_path = '../trained_models/simpler_CNN.hdf5'
faces, emotions = load_data(data_path)
faces = preprocess_input(faces)
num_classes = emotions.shape[1]
image_size = faces.shape[1:]
batch_size = 128
num_epochs = 1000

model = simple_CNN(image_size, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
                                        metrics=['accuracy'])
csv_logger = CSVLogger('training.log')
early_stop = EarlyStopping('val_acc',patience=200,verbose=1)
model_checkpoint = ModelCheckpoint(model_save_path,
                                    'val_acc', verbose=1,
                                    save_best_only=True)

model_callbacks = [early_stop, model_checkpoint, csv_logger]

#keras bug 
K.get_session().run(tf.global_variables_initializer())
model.fit(faces,emotions,batch_size,num_epochs,verbose=1,
                                    callbacks=model_callbacks,
                                    validation_split=.1,
                                    shuffle=True)
