import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU
from thermography.classification.models.base_net import BaseNet

class ThermoNet(BaseNet):
    def __init__(self, image_shape: np.ndarray, num_classes: int, keep_prob: float):
        super().__init__(image_shape=image_shape, num_classes=num_classes, name="ThermoNet")
        self.keep_probability = keep_prob
        self.model = self.create()

    def create(self):
        inputs = Input(shape=self.image_shape)

        # Convolutional layer 1
        x = Conv2D(8, (5, 5), activation='relu', padding='same', name='conv_1_0')(inputs)
        x = MaxPooling2D((4, 4), name='max_pool_1')(x)

        # Convolutional layer 2
        x = Conv2D(16, (5, 5), activation='relu', padding='same', name='conv_2_0')(x)
        x = MaxPooling2D((4, 4), name='max_pool_2')(x)

        # Flatten
        x = Flatten(name='flattened')(x)

        # Fully connected layer 1
        x = Dense(256, activation='relu', name='fc_1')(x)

        # Dropout
        x = Dropout(1 - self.keep_probability, name='dropout_1')(x)

        # Fully connected layer 2
        outputs = Dense(self.num_classes, activation='softmax', name='logits')(x)

        model = Model(inputs=inputs, outputs=outputs, name=self.name)
        return model

