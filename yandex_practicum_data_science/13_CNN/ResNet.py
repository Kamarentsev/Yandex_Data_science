#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

def load_train(path):
    train_datagen = ImageDataGenerator(
        rescale=1/255.,
        horizontal_flip=True,
#         vertical_flip=True,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rotation_range=90
    )
    train_datagen_flow = train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training',
        seed=12345,
    )

    return train_datagen_flow


def create_model(input_shape):
    backbone = ResNet50(
        input_shape=input_shape,
        weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        include_top=False,
        )

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(128, activation='relu'))
    model.add(Dense(12, activation='softmax'))

    # Используем Adam с learning rate 0.001
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model


def train_model(
        model,
        train_data,
        test_data,
        batch_size=None,
        epochs=7,
        steps_per_epoch=None,
        validation_steps=None,
):
    model.fit(
        train_data,
        validation_data=test_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
        shuffle=True,
    )

    return model


# In[ ]:




