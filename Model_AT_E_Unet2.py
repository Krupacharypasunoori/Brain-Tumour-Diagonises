from keras import Input, Model
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, LayerNormalization, MultiHeadAttention
from keras.applications import EfficientNetB7
from numpy import concatenate
import numpy as np
import cv2 as cv
from tensorflow.keras.layers import Add
from Evaluation_All import seg_evaluation


def attention_block(x, head_size, num_heads):
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
    attention = LayerNormalization()(attention)
    return Add()([x, attention])


def trans_efficient_unet2_plus(input_shape=(256, 256, 3), sol= 5, head_size=64, num_heads=4):
    inputs = Input(shape=input_shape)

    # Encoder with EfficientNetB7
    efficientnet = EfficientNetB7(include_top=False, input_shape=input_shape)(inputs)
    enc1 = efficientnet.get_layer("block2a_activation").output  # output from EfficientNet block
    enc2 = efficientnet.get_layer("block3a_activation").output
    enc3 = efficientnet.get_layer("block4a_activation").output
    enc4 = efficientnet.get_layer("block5a_activation").output
    enc5 = efficientnet.get_layer("top_activation").output

    # Apply Attention on Encoded Layers
    enc1 = attention_block(enc1, head_size, num_heads)
    enc2 = attention_block(enc2, head_size, num_heads)
    enc3 = attention_block(enc3, head_size, num_heads)
    enc4 = attention_block(enc4, head_size, num_heads)
    enc5 = attention_block(enc5, head_size, num_heads)

    # Decoder
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(enc5)
    up6 = concatenate([up6, enc4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, enc3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, enc2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, enc1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(sol, 3, activation='relu', padding='same')(conv9)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Usage example:
def Model_AT_E_Unet2(train_data, train_target, sol=None):
    if sol is None:
        sol = [5, 0.01, 100]
    IMG_SIZE = 256
    seed = 42
    np.random.seed(seed)

    X_train = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    Y_train = np.zeros((train_target.shape[0], IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    for i in range(train_data.shape[0]):
        Temp = cv.resize(train_data[i, :], (IMG_SIZE, IMG_SIZE))
        X_train[i, :, :, :] = Temp.reshape((IMG_SIZE, IMG_SIZE, 3))

    for i in range(train_target.shape[0]):
        Temp = cv.resize(train_target[i, :], (IMG_SIZE, IMG_SIZE))
        Temp = Temp.reshape((IMG_SIZE, IMG_SIZE, 3))
        for j in range(Temp.shape[0]):
            for k in range(Temp.shape[1]):
                Temp[j, k] = 1 if Temp[j, k] >= 0.5 else 0
        Y_train[i, :, :, :] = Temp

    model = trans_efficient_unet2_plus(input_shape=(IMG_SIZE, IMG_SIZE, 3), sol=sol[0])
    model.summary()
    model.fit(X_train, Y_train, epochs=sol[1],steps_per_epoch=sol[2], batch_size=8, validation_split=0.1)
    pred_img = model.predict(X_train)
    ret_img = pred_img[:, :, :, 0]
    Eval = seg_evaluation(ret_img, train_target)
    return Eval, ret_img
