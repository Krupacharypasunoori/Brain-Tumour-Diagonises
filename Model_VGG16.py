from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Sequential
import cv2 as cv
import numpy as np

from Classificaltion_Evaluation import ClassificationEvaluation


def VGG_16(num_of_class=None):
    model = Sequential()
    model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=num_of_class, activation="softmax"))
    return model


def Model_VGG16(train_data, train_target, test_data, test_target):
    IMG_SIZE = [32, 32, 3]

    Feat1 = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat1[i] = cv.resize(train_data[i], (IMG_SIZE[1], IMG_SIZE[0]))

    Feat2 = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Feat2[i] = cv.resize(test_data[i], (IMG_SIZE[1], IMG_SIZE[0]))

    model = VGG_16(num_of_class=train_target.shape[1])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x=Feat1, y=train_target, epochs=18, steps_per_epoch=1)

    pred = model.predict(Feat2)

    min = np.min(pred)
    max = np.max(pred)
    avg = (min + max) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = ClassificationEvaluation(pred, test_target)  # Ensure evaluation function is defined
    return np.asarray(Eval)
