import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import scipy.io   
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
 

train_dataset=scipy.io.loadmat('train_32x32.mat')
test_dataset=scipy.io.loadmat('test_32x32.mat')

x_train =train_dataset['X']
y_train=train_dataset['y']
x_test=test_dataset['X']
y_test=test_dataset['y']
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
 

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train,axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)
 

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
 
weights = 1e-4
modelseq = Sequential()
modelseq.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weights), input_shape=x_train.shape[1:]))
modelseq.add(Activation('elu'))
modelseq.add(BatchNormalization())
modelseq.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weights)))
modelseq.add(Activation('elu'))
modelseq.add(BatchNormalization())
modelseq.add(MaxPooling2D(pool_size=(1,1),dim_ordering="tf"))
modelseq.add(Dropout(0.2))
 
modelseq.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weights)))
modelseq.add(Activation('elu'))
modelseq.add(BatchNormalization())
modelseq.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weights)))
modelseq.add(Activation('elu'))
modelseq.add(BatchNormalization())
modelseq.add(MaxPooling2D(pool_size=(1,1),dim_ordering="tf"))
modelseq.add(Dropout(0.3))
 
modelseq.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weights)))
modelseq.add(Activation('elu'))
modelseq.add(BatchNormalization())
modelseq.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weights)))
modelseq.add(Activation('elu'))
modelseq.add(BatchNormalization())
modelseq.add(MaxPooling2D(pool_size=(1,1),dim_ordering="tf"))
modelseq.add(Dropout(0.4))
 
modelseq.add(Flatten())
modelseq.add(Dense(10,activation='softmax'))
 
modelseq.summary()
 

datagenerator = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagenerator.fit(x_train)
 

opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
modelseq.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
modelseq.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),\
                    steps_per_epoch=x_train.shape[0] // batch_size,epochs=125,\
                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)])
#save HDF5 file
HDF5 = modelseq.to_json()
with open('modelseq.json', 'w') as json_file:
    json_file.write(HDF5)
modelseq.save_weights('modelseq.h5') 
 
#testing
scores = modelseq.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('\n resultst: %.3f loss: %.3f' % (scores[1]*100,scores[0])
