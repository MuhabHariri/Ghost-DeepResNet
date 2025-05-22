import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

class ECALayer(layers.Layer):
    """
    Efficient Channel Attention (ECA) Layer.
    This layer applies channel-wise attention using a 1D convolution over 
    global average pooled features. Kernel size is adaptively determined.
    """

    def __init__(self, channels, gamma=2, b=1, activation='sigmoid', **kwargs):
        super(ECALayer, self).__init__(**kwargs)
        self.channels = channels
        self.gamma = gamma
        self.b = b
        self.activation = activation

        #Calculate adaptive kernel size (odd number)
        t = abs((np.log2(channels) + b) / gamma)
        k = int(t) if int(t) % 2 else int(t) + 1

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.conv = layers.Conv1D(
            filters=1,
            kernel_size=k,
            padding='same',
            use_bias=False,
            activation=activation
        )

    def call(self, inputs):
        y = self.avg_pool(inputs)                        
        y = tf.expand_dims(y, axis=-1)                    
        y = self.conv(y)                                  
        y = tf.squeeze(y, axis=-1)                        
        y = tf.reshape(y, [-1, 1, 1, self.channels])       
        return inputs * y

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'gamma': self.gamma,
            'b': self.b,
            'activation': self.activation,
        })
        return config



class WeightedResidualConnection(layers.Layer):
    """
    y = x + α · residual

    A learnable scalar (α) blends the residual branch into the main path.

    """

    def __init__(self, init_value: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.init_value = float(init_value)


    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.init_value),
            trainable=True,
            dtype=tf.float32,
        )
        super().build(input_shape)


    def call(self, x, residual):
        return layers.add([x, residual * self.alpha])

 
    def get_config(self):
        config = super().get_config()
        config.update({"init_value": self.init_value})
        return config









