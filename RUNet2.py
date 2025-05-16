from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from numpy import concatenate
import numpy as np
import random as rn
import cv2 as cv
from Evaluation_All import seg_evaluation
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Cropping2D
from keras.models import Model


def crop_and_concat(target, reference):
    """ Crop `target` tensor to match the shape of `reference` and concatenate. """
    target_shape = tf.shape(target)
    reference_shape = tf.shape(reference)

    crop_h = (target_shape[1] - reference_shape[1]) // 2
    crop_w = (target_shape[2] - reference_shape[2]) // 2

    if crop_h > 0 or crop_w > 0:
        target = Cropping2D(((crop_h, crop_h), (crop_w, crop_w)))(target)

    return concatenate([target, reference], axis=3)


def runet8_model(input_shape=(512, 512, 3)):
    inputs = Input(shape=input_shape)

    # Encoder Path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Decoder Path
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    up6 = crop_and_concat(up6, conv4)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = crop_and_concat(up7, conv3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = crop_and_concat(up8, conv2)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = crop_and_concat(up9, conv1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def RUNet2(train_data, train_target):
    # Set parameters
    IMG_SIZE = 256
    seed = 42
    rn.seed(seed)
    np.random.seed(seed)

    # Prepare training data
    X_train = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Y_train = np.zeros((train_target.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)

    # Resize input images
    for i in range(train_data.shape[0]):
        X_train[i] = cv.resize(train_data[i, :], (IMG_SIZE, IMG_SIZE))

    # Resize and threshold target images
    for i in range(train_target.shape[0]):
        Temp = cv.resize(train_target[i, :], (IMG_SIZE, IMG_SIZE))

        # Convert to grayscale if necessary
        if Temp.ndim == 3:
            Temp = cv.cvtColor(Temp, cv.COLOR_BGR2GRAY)

        # Thresholding: Convert to binary mask
        Temp = (Temp >= 128).astype(np.uint8)  # Assuming 8-bit grayscale images

        Y_train[i, :, :, 0] = Temp  # Store in single channel

    # Build and train model
    model = runet8_model()
    model.summary()
    model.fit(X_train, Y_train, epochs=10, batch_size=8, verbose=1)  # Adjust epochs and batch size as needed

    # Predict
    pred_img = model.predict(X_train)
    ret_img = pred_img[:, :, :, 0]  # Extract single channel

    # Evaluate
    Eval = seg_evaluation(ret_img, Y_train)
    return Eval, ret_img
