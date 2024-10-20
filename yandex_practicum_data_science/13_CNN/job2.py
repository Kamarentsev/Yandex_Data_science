from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import AveragePooling2D,Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping

def load_train(path='/datasets/fruits_small/'): 
	datagen = keras.preprocessing.image.ImageDataGenerator(
		validation_split=0.25, 
        rescale=1./255.#,
        #horizontal_flip=True,
        #vertical_flip=True
		#rotation_range=90,
		#width_shift_range=0.2,
		#height_shift_range=0.2
	)
	train_datagen_flow = datagen.flow_from_directory(
		path,
		target_size=(150, 150),
		batch_size=32,
        class_mode='sparse',
		subset='training',
		seed=12345)
	return train_datagen_flow

def create_model(input_shape): 
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    model.add(AveragePooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(12, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=0.001) 
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc']) 
	
    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=10, steps_per_epoch=None, validation_steps=None):
	#early_stopping = EarlyStopping(monitor='val_loss', patience=10)
	model.fit(
		train_data,
        validation_data=test_data,
		batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2, 
		#callbacks=[early_stopping]
		epochs=epochs,
		shuffle=True
		)
	return model