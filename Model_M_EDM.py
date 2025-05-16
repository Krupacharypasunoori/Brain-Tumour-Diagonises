import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

from Classificaltion_Evaluation import ClassificationEvaluation


def get_multiscale_efficientdet_backbone():
    # Load EfficientNetB0 as the base model
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(512, 512, 3),
        weights='imagenet'
    )

    # Obtain the relevant layers for multiscale features
    outputs = [base_model.get_layer('block2a_expand_relu').output,  # Low-level features
               base_model.get_layer('block3a_expand_relu').output,  # Intermediate features
               base_model.get_layer('top_activation').output]  # High-level features

    backbone = keras.Model(inputs=base_model.input, outputs=outputs)
    return backbone


# Modify segmentation head for multiscale outputs
def segmentation_head_multiscale(x, num_classes):
    # Process each feature map from different scales
    x1 = layers.Conv2D(256, (3, 3), padding='same')(x[0])
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)

    x2 = layers.Conv2D(256, (3, 3), padding='same')(x[1])
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)

    x3 = layers.Conv2D(256, (3, 3), padding='same')(x[2])
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)

    # Combine the feature maps from different scales
    x = layers.Concatenate()([x1, x2, x3])

    # Upsampling and final segmentation layer
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)
    x = layers.UpSampling2D(size=(4, 4))(x)  # Upsample to original image size

    return x


def build_multiscale_efficientdet_segmentation_model(num_classes):
    inputs = keras.Input(shape=(512, 512, 3))
    backbone = get_multiscale_efficientdet_backbone()
    backbone_outputs = backbone(inputs)

    segmentation_output = segmentation_head_multiscale(backbone_outputs, num_classes)

    model = keras.Model(inputs=inputs, outputs=segmentation_output)
    return model


def Model_M_EDM(Image, Target):
    Images = []
    for i in range(len(Image)):
        Im = cv2.cvtColor(Image[i], cv2.COLOR_GRAY2RGB)
        Images.append(Im)

    Images = np.asarray(Images)

    num_classes = 3  # Example for PASCAL VOC dataset

    model = build_multiscale_efficientdet_segmentation_model(num_classes)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    results = model.predict(Images)
    Eval = ClassificationEvaluation(Target[:, :, :, 0][0, :results.shape[1], :results.shape[1]], results[0, :, :, 0])
    return Eval, results
