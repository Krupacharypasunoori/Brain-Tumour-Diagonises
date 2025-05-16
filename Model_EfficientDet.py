import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

from Classificaltion_Evaluation import ClassificationEvaluation


def get_efficientdet_d0_backbone():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(512, 512, 3),
        weights='imagenet'
    )
    backbone = keras.Model(inputs=base_model.input, outputs=base_model.get_layer('top_activation').output)
    return backbone


# Draw bounding boxes on the image
def draw_boxes(image, results):
    for box, class_id, score in results:
        y1, x1, y2, x2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"Class {class_id}: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image


def segmentation_head(x, num_classes):
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)
    x = layers.UpSampling2D(size=(4, 4))(x)  # Upsample to original image size
    return x


def build_efficientdet_segmentation_model(num_classes):
    inputs = keras.Input(shape=(512, 512, 3))
    backbone = get_efficientdet_d0_backbone()
    backbone_output = backbone(inputs)
    segmentation_output = segmentation_head(backbone_output, num_classes)

    model = keras.Model(inputs=inputs, outputs=segmentation_output)
    return model


def Model_EfficientDet(Image, Target):
    Images = []
    for i in range(len(Image)):
        Im = cv2.cvtColor(Image[i], cv2.COLOR_GRAY2RGB)
        Images.append(Im)

    Images = np.asarray(Images)

    num_classes = 3  # Example for PASCAL VOC dataset

    model = build_efficientdet_segmentation_model(num_classes)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    results = model.predict(Images)
    Eval = ClassificationEvaluation(Target[:, :, :, 0][0, :results.shape[1], :results.shape[1]], results[0, :, :, 0])
    return Eval, results
