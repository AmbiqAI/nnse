import tensorflow as tf

class SeparableTransposeConv2D(tf.keras.layers.Layer):
    """ Transpose convolutional layer"""
    def __init__(
            self,
            filters,
            kernel_size,
            num_channels_in=1,
            activation=None,
            **kwargs):
        super(SeparableTransposeConv2D, self).__init__(**kwargs)
        kernel_initializer = "he_normal" if activation in ('relu', 'relu6') else "glorot_uniform"
        len_filter_freq = kernel_size[1]
        self.upsampling2= tf.keras.layers.UpSampling2D(
            size=(1,2),
            interpolation='nearest'
        )
        self.padding = tf.keras.layers.ZeroPadding2D(
            padding=((0, 0),(len_filter_freq-1,len_filter_freq-1))
        )
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
            padding='same',
            use_bias=True,
            kernel_initializer=kernel_initializer,
            activation=activation)

    def call(self, inputs):
        """ Forward pass """
        # input shape = (B, T, F, C)
        # inputs_up = updsampling_by_2(inputs)
        inputs_up = self.upsampling2(inputs)
        inputs_up = inputs_up[:,:,:-1,:]
        inputs_up = self.padding(inputs_up)
        outputs = self.depthwise(inputs_up)
        outputs = self.pointwise(outputs)
        return outputs

class TransposeConv2D(tf.keras.layers.Layer):
    """ Transpose convolutional layer"""
    def __init__(
            self,
            filters,
            kernel_size,
            activation=None,
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

    def call(self, inputs):
        """ Forward pass """
        # input shape = (B, T, F, C)
        # inputs_up = updsampling_by_2(inputs)
        inputs_up = self.upsampling2(inputs)
        inputs_up = inputs_up[:,:,:-1,:]
        inputs_up = self.padding(inputs_up)
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

