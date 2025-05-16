import numpy as np
from keras import Input, Model
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.layers import *
import cv2 as cv
from keras.src.optimizers import Adam

from Evaluation_All import seg_evaluation


def yolo_v8(input_shape, num_classes):
    inputs = Input(input_shape)

    # Backbone: Simplified version, typically a CSPDarknet in YOLOv8
    # Conv Block 1
    x = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Conv Block 2
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Conv Block 3
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # YOLOv8 specific layers (detection head)
    # Simplified detection head: We use 1x1 convolution to predict class scores, objectness, and bounding boxes
    detection_head = Conv2D(num_classes + 5, 1, activation='linear', padding='same')(
        x)  # 5 for 4 bbox coords + objectness

    # Output
    outputs = detection_head

    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model


def YoloV8(Images, Groundtruth):
    input_shape = (512, 512, 3)  # YOLO typically works with 416x416 image size
    num_classes = 2
    IMG_SIZE = 512
    learnperc = round(Images.shape[0] * 0.75)
    train_data = Images[:learnperc]
    train_target = Groundtruth[:learnperc]
    test_data = Images[learnperc:]
    test_target = Groundtruth[learnperc:]

    X_train = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Y_train = np.zeros((train_target.shape[0], IMG_SIZE, IMG_SIZE, num_classes + 5), dtype=np.uint8)
    X_test = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Y_test = np.zeros((test_target.shape[0], IMG_SIZE, IMG_SIZE, num_classes + 5), dtype=np.uint8)

    for i in range(train_data.shape[0]):
        Temp = cv.resize(train_data[i, :], (IMG_SIZE, IMG_SIZE))
        X_train[i, :, :, :] = Temp.reshape((IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_target.shape[0]):
        Temp = cv.resize(train_target[i, :], (IMG_SIZE, IMG_SIZE))
        Temp = np.resize(Temp, (IMG_SIZE, IMG_SIZE, num_classes + 5)).astype('uint8')
        Y_train[i, :, :, :] = Temp

    for i in range(test_data.shape[0]):
        Temp = cv.resize(test_data[i, :], (IMG_SIZE, IMG_SIZE))
        X_test[i, :, :, :] = Temp.reshape((IMG_SIZE, IMG_SIZE, 3))

    model = yolo_v8(input_shape, num_classes)
    model.summary()
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('yolov8_model.keras', verbose=1, save_best_only=True)

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=2,
              callbacks=[earlystopper, checkpointer])

    pred_img = model.predict(X_test)
    # You would typically do some post-processing on the predictions
    Eval = seg_evaluation(pred_img, Y_test)
    return Eval, pred_img
