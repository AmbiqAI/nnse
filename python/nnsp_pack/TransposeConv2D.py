import tensorflow as tf
from .dynamnic_tanh import dynamnic_tanh
class SeparableTransposeConv2D(tf.keras.layers.Layer):
    """ Transpose convolutional layer"""
    def __init__(
            self,
            filters,
            kernel_size,
            num_channels_in=1,
            activation=None,
            batch_size=1,
            time_steps=1,
            normalization_layer=None,
            **kwargs):
        super(SeparableTransposeConv2D, self).__init__(**kwargs)

        kernel_initializer = "he_normal" if activation in ('relu', 'relu6') else "glorot_uniform"
        len_filter_freq = kernel_size[1]
        self.kernel_size_time = kernel_size[0]
        self.upsampling2= tf.keras.layers.UpSampling2D(
            size=(1,2),
            interpolation='nearest'
        )
        self.padding = tf.keras.layers.ZeroPadding2D(
            padding=((0, 0),(len_filter_freq-1,len_filter_freq-1))
        )
        
        if normalization_layer is None:
            use_bias=True
        else:
            use_bias=False
        self.depthwise = tf.keras.layers.Conv2D(
            filters=num_channels_in,
            kernel_size=kernel_size,
            padding='valid',
            groups=num_channels_in,
            use_bias=False,
            kernel_initializer=kernel_initializer)
        self.pointwise = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            use_bias=use_bias,
            kernel_initializer=kernel_initializer)
        if normalization_layer == 'layernorm':
            self.normalization_layer = tf.keras.layers.LayerNormalization(axis=[2,3])
        elif normalization_layer == 'batchnorm':
            self.normalization_layer = tf.keras.layers.BatchNormalization()
        else:
            self.normalization_layer = None
        
        if activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == 'relu':
            self.activation = tf.nn.relu
        elif activation == 'sigmoid':
            self.activation = tf.nn.sigmoid
        else:
            self.activation = None
        self.zeros=tf.zeros(
            (batch_size,
             time_steps + self.kernel_size_time - 1,
             len_filter_freq-1,
             num_channels_in)) # x2 since channel is doubled after concatenation

    def call(self, inputs):
        """ Forward pass """
        # input shape = (B, T, F, C)
        # inputs_up = updsampling_by_2(inputs)
        inputs_up = self.upsampling2(inputs)
        inputs_up = inputs_up[:,:,:-1,:]
        inputs_up = tf.concat(
            [self.zeros, inputs_up, self.zeros], axis=-2)

        # inputs_up = self.padding(inputs_up)
        outputs = self.depthwise(inputs_up)

        outputs = self.pointwise(outputs)
        if self.normalization_layer is not None:
            outputs = self.normalization_layer(outputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

class TransposeConv2D(tf.keras.layers.Layer):
    """ Transpose convolutional layer"""
    def __init__(
            self,
            filters,
            kernel_size,
            num_channels_in=1,
            activation=None,
            time_steps=2,
            num_bathces=1,
            **kwargs):
        super(TransposeConv2D, self).__init__(**kwargs)
        kernel_initializer = "he_normal" if activation in ('relu', 'relu6') else "glorot_uniform"

        len_filter_freq = kernel_size[1]
        self.upsampling2= tf.keras.layers.UpSampling2D(
            size=(1,2),
            interpolation='nearest'
        )
        self.padding = tf.keras.layers.ZeroPadding2D(
            padding=((0, 0),(len_filter_freq-1,len_filter_freq-1))
        )
        self.deconv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='valid',
            kernel_initializer=kernel_initializer,
            activation=activation)
        self.zeros=tf.zeros((num_bathces,time_steps,len_filter_freq-1,num_channels_in))
    def call(self, inputs):
        """ Forward pass """
        # input shape = (B, T, F, C)
        # inputs_up = updsampling_by_2(inputs)
        inputs_up = self.upsampling2(inputs)
        inputs_up = inputs_up[:,:,:-1,:]
        inputs_up = tf.concat(
            [self.zeros, inputs_up, self.zeros], axis=-2)
        # inputs_up = self.padding(inputs_up)
        outputs = self.deconv(inputs_up)
        return outputs

def updsampling_by_2(inputs):
    """ upsample the input
    Args:
        inputs: (B, T, F, C)
        Returns:
        inputs_up: (B, T, 2*F, C)
        example:
        inputs = tf.constant(
            [[[1,2,3],
              [1,2,3]]
                ])
        inputs1 = tf.constant(
            [[[4,5,6],
              [4,5,6]]
                ])
        inputs = tf.concat([inputs, inputs1], axis=-1)
        inputs = tf.expand_dims(inputs, axis=-1)
        print(inputs[0,:,:,0])
        print(inputs[0,:,:,1])
        >>
            tf.Tensor(
            [[1 2 3]
             [1 2 3]], shape=(2, 3), dtype=int32)
            tf.Tensor(
            [[4 5 6]
             [4 5 6]], shape=(2, 3), dtype=int32)
        outputs = updsampling_by_2(inputs)
        print(outputs[0,:,:,0])
        print(outputs[0,:,:,1])
        >>
            tf.Tensor(
            [[1 0 2 0 3]
             [1 0 2 0 3]], shape=(2, 5), dtype=int32)
            tf.Tensor(
            [[4 0 5 0 6]
             [4 0 5 0 6]], shape=(2, 5), dtype=int32)
        """
    shape = tf.shape(inputs)
    B = shape[0]
    T = shape[1]
    F = shape[2]
    C = shape[3]
    # copy = tf.zeros_like(inputs)
    # inputs_up = tf.concat([inputs, copy], axis=1)
    if 1: 
        inputs_up = tf.pad(inputs, [[0, 0], [T, 0], [0, 0], [0, 0]])
        inputs_up = tf.transpose(inputs_up, perm=[0, 2, 1, 3])
        inputs_up = tf.reshape(inputs_up, (B, -1, T, C))
        inputs_up = tf.transpose(inputs_up, perm=[0, 2, 1, 3])
    else: 
        pass
    inputs_up = inputs_up[:,:,:-1,:]
    return inputs_up

if __name__ == '__main__':
    inputs = tf.constant(
        [[[1,2,3],
          [1,2,3]]],
        dtype=tf.float32)
    inputs1 = tf.constant(
        [[[4,5,6],
          [4,5,6]]],
        dtype=tf.float32)
    inputs = tf.expand_dims(inputs, axis=-1)

    inputs1 = tf.expand_dims(inputs1, axis=-1)
    inputs = tf.concat([inputs, inputs1], axis=-1)
    print(inputs[0,:,:,0])
    print(inputs[0,:,:,1])
    deconv = TransposeConv2D(1,(2,3))
    deconv(inputs)

    deconv.deconv.kernel.assign(deconv.deconv.kernel*0+1)
    inputs_up = deconv(inputs)
    print(inputs_up[0,:,:,0])

