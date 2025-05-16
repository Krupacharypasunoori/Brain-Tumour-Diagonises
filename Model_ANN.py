from keras import Sequential
import numpy as np
from keras.layers import Dropout, Dense, Flatten
from Classificaltion_Evaluation import ClassificationEvaluation


def Model_ANN(train_data, train_target, test_data, test_target):
    print('ANN')
    IMG_SIZE = 28
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE * IMG_SIZE))
    for i in range(train_data.shape[0]):
        Train_X[i] = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE,))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE * IMG_SIZE))
    for i in range(test_data.shape[0]):
        Test_X[i] = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE,))

    X_train = Train_X.astype('float32') / 255
    X_test = Test_X.astype('float32') / 255

    model = Sequential([
        Dense(512, activation='relu', input_shape=(IMG_SIZE * IMG_SIZE,)),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(train_target.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, train_target, epochs=15, batch_size=64, validation_split=0.1)
    pred = model.predict(X_test)

    min_val = np.min(pred)
    max_val = np.max(pred)
    avg = (min_val + max_val) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0

    Eval = ClassificationEvaluation(test_target, pred)
    return Eval
