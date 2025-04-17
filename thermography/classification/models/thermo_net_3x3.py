# import numpy as np
# import tensorflow as tf
# from thermography.classification.utils.operations import *
# from thermography.classification.models.base_net import BaseNet
# from tensorflow import keras
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU, Activation
# class ThermoNet3x3(BaseNet):
#     def __init__(self,  image_shape: tuple, num_classes: int, keep_prob: float):
#         # 调用 BaseNet 的构造函数，并传递必要的参数
#         super().__init__(image_shape=image_shape, num_classes=num_classes, name="ThermoNet3x3")
#         self.keep_probability = keep_prob
#         self.model = self.create()
#
#     def create(self):
#         inputs = Input(shape=self.image_shape)
#
#         # Convolutional layer 1
#         x = Conv2D(8, (5, 5), activation='relu', padding='same', name='conv_1_0')(inputs)
#         x = MaxPooling2D((4, 4), name='max_pool_1')(x)
#
#         # Convolutional layer 2
#         x = Conv2D(16, (5, 5), activation='relu', padding='same', name='conv_2_0')(x)
#         x = MaxPooling2D((4, 4), name='max_pool_2')(x)
#
#         # Flatten
#         x = Flatten(name='flattened')(x)
#
#         # Fully connected layer 1
#         x = Dense(256, activation='relu', name='fc_1')(x)
#
#         # Dropout
#         x = Dropout(1 - self.keep_probability, name='dropout_1')(x)
#
#
#         # Fully connected layer 2
#         logit = Dense(self.num_classes, name='logits')(x)
#
#         # 应用 softmax 激活函数
#         outputs = Activation('softmax', name='softmax')(logit)
#
#         self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
#
#         return self.model
#
#
#     @property
#     def logits(self):
#         """直接访问 logits 层输出"""
#         return self.logits_layer.output
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

        # 初始化特征图尺寸跟踪（用于计算 Flatten 后的维度）
        self.current_shape = np.array(image_shape[:2])  # [H, W]

        # --- 定义所有层，严格对齐原代码结构 ---
        # 卷积层 1: 5x5 卷积核，输出通道 8 → 池化 2x2
        self.conv1 = Conv2D(8, (5, 5), activation='relu', padding='same', name='conv_1_0')
        self.maxpool1 = MaxPooling2D((2, 2), name='max_pool_1')  # 池化窗口 2x2
        self.current_shape = self.current_shape // 2  # 更新尺寸

        # 卷积层 2: 3x3 卷积核，输出通道 16 → 池化 4x4
        self.conv2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_2_0')
        self.maxpool2 = MaxPooling2D((4, 4), name='max_pool_2')  # 池化窗口 4x4
        self.current_shape = self.current_shape // 4  # 更新尺寸

        # 卷积层 3: 3x3 卷积核，输出通道 32 → 池化 2x2
        self.conv3 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_3_0')
        self.maxpool3 = MaxPooling2D((2, 2), name='max_pool_3')  # 池化窗口 2x2
        self.current_shape = self.current_shape // 2  # 更新尺寸

        # Dropout 1（卷积层后）
        self.dropout1 = Dropout(1 - self.keep_prob, name='drop_out_1')

        # Flatten 层（计算维度）
        self.flatten = Flatten(name='flattened')
        self.flatten_dim = np.prod(self.current_shape) * 32  # 32 是最后一个卷积层的通道数

        # 全连接层 1: 输入 → 256 单元
        self.fc1 = Dense(256, activation='relu', name='fc_1')

        # 全连接层 2: 256 → 32 单元
        self.fc2 = Dense(32, activation='relu', name='fc_2')

        # Dropout 2（全连接层后）
        self.dropout2 = Dropout(1 - self.keep_prob, name='drop_out_2')

        # 输出层（logits）
        self.logits_layer = Dense(num_classes, name='readout')

    def call(self, inputs, training=False):
        """严格对齐原代码的前向传播逻辑"""
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
        """显式构建模型（用于维度计算）"""
        super().build(input_shape)

# @property
    # def logits(self):
    #     """直接访问 logits 层输出"""
    #     return self.logits



# class ThermoNet3x3(BaseNet):
#     def __init__(self, image_shape: tuple, num_classes: int, keep_prob: float, **kwargs):
#         # 将 tuple 转换为 np.ndarray，避免类型冲突
#         image_shape_np = np.array(image_shape, dtype=np.int32)
#         # 通过父类属性设置，避免触发子类逻辑
#         super().__init__(image_shape=image_shape_np, num_classes=num_classes, **kwargs)
#         self.keep_prob = keep_prob
#         self.build(image_shape_np)
#
#     def build(self, input_shape):
#         inputs = Input(shape=input_shape)
#         x = Conv2D(8, (5,5), activation='relu', padding='same')(inputs)
#         x = MaxPooling2D((4,4))(x)
#         x = Conv2D(16, (5,5), activation='relu', padding='same')(x)
#         x = MaxPooling2D((4,4))(x)
#         x = Flatten()(x)
#         x = Dense(256, activation='relu')(x)
#         x = Dropout(self.keep_prob)(x)
#         logits = Dense(self.num_classes)(x)
#         outputs = Activation('softmax')(logits)
#         self.model = Model(inputs=inputs, outputs=outputs)
#
#     @property
#     def logits(self):
#         return self.logits_layer.output