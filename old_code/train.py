from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adadelta
#from models import simple_CNN
from models import attention_CNN
from utils import load_data, preprocess_input

data_path = '../datasets/fer2013/fer2013.csv'
model_save_path = '../trained_models/emotion_classifier_attention_v2.hdf5'
faces, emotions = load_data(data_path)
faces = preprocess_input(faces)
num_classes = emotions.shape[1]
image_size = faces.shape[1:]
batch_size = 32
num_epochs = 1000

#model = simple_CNN(image_size, num_classes)
model = attention_CNN(image_size, num_classes)
Adadelta(lr=1.0, decay=.01)
model.compile(optimizer='adam', loss='categorical_crossentropy',
                                        metrics=['accuracy'])
csv_logger = CSVLogger('attention_training.log', append=False)
model_checkpoint = ModelCheckpoint(model_save_path,
                                    'val_acc', verbose=1,
                                    save_best_only=True)

model_callbacks = [model_checkpoint, csv_logger]

model.fit(faces,emotions,batch_size,num_epochs,verbose=1,
                                    callbacks=model_callbacks,
                                    validation_split=.1,
                                    shuffle=True)
