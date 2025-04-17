
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model ,Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from thermography.classification.models.base_net import BaseNet
#
#
class ThermoNet3x3(BaseNet):
    def __init__(self, image_shape: np.ndarray, num_classes: int, keep_prob: float):
        super().__init__(image_shape=image_shape, num_classes=num_classes, name="ThermoNet3x3")
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.keep_prob = keep_prob

        # Initialize feature map size tracking
        self.current_shape = np.array(image_shape[:2])  # [H, W]

        self.conv1 = Conv2D(8, (5, 5), activation='relu', padding='same', name='conv_1_0')
        self.maxpool1 = MaxPooling2D((2, 2), name='max_pool_1')  # 池化窗口 2x2
        self.current_shape = self.current_shape // 2  # 更新尺寸

        self.conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_2_0')
        self.maxpool2 = MaxPooling2D((4, 4), name='max_pool_2')  # 池化窗口 4x4
        self.current_shape = self.current_shape // 4  # 更新尺寸

        self.conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_3_0')
        self.maxpool3 = MaxPooling2D((2, 2), name='max_pool_3')  # 池化窗口 2x2
        self.current_shape = self.current_shape // 2  # 更新尺寸

        self.dropout1 = Dropout(1 - self.keep_prob, name='drop_out_1')

        self.flatten = Flatten(name='flattened')
        self.flatten_dim = np.prod(self.current_shape) * 32  # 32 是最后一个卷积层的通道数

        self.fc1 = Dense(256, activation='relu', name='fc_1')

        self.fc2 = Dense(32, activation='relu', name='fc_2')

        self.dropout2 = Dropout(1 - self.keep_prob, name='drop_out_2')

        self.logits_layer = Dense(num_classes, name='readout')

    def call(self, inputs, training=False):
        """Creates the ThermoNet3x3 computation graph consisting of three convolutional layers followed by three fully connected layers"""
        # 卷积层 1
        x = self.conv1(inputs)
        x = self.maxpool1(x)

        # 卷积层 2
        x = self.conv2(x)
        x = self.maxpool2(x)

        # 卷积层 3
        x = self.conv3(x)
        x = self.maxpool3(x)

        # Dropout 1
        x = self.dropout1(x, training=training)

        # Flatten（确保维度计算正确）
        x = self.flatten(x)

        # 全连接层 1
        x = self.fc1(x)

        # 全连接层 2
        x = self.fc2(x)

        # Dropout 2
        x = self.dropout2(x, training=training)

        # 输出 logits（不包含 softmax）
        logits = self.logits_layer(x)
        return logits

    def build(self, input_shape):
        super().build(input_shape)





