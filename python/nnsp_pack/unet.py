import tensorflow as tf
# @tf.keras.saving.register_keras_serializable()
LEN_TIME_KERNEL = 3

class SliceLayer(tf.keras.layers.Layer):
    """ Slice layer"""
    def __init__(
            self,
            kernel_size_time=3,
            is_causal=True,
            **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.kernel_size_time = kernel_size_time
        self.is_causal = is_causal
    def call(self, inputs):
        """ Forward pass"""
        if self.is_causal:
            outputs=inputs[:,:-(self.kernel_size_time-1),:,:] # remove the last (kernel_size_time-1) frames
        else:
            outputs=inputs[:,self.kernel_size_time-1:,:,:]
        return outputs

class SeparableConv2D(tf.keras.layers.Layer):
    """ Separable convolutional layer"""
    def __init__(
            self,
            filters,
            kernel_size,
            strides=(1, 1),
            num_channels_in=1,
            activation=None,
            **kwargs):
        super(SeparableConv2D, self).__init__(**kwargs)
        self.depthwise = tf.keras.layers.Conv2D(
            filters=num_channels_in,
            kernel_size = kernel_size,
            strides=strides,
            groups = num_channels_in,
            use_bias=False,
            kernel_initializer='he_normal')
        self.pointwise = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            use_bias=True,
            kernel_initializer='he_normal',
            activation=activation)

    def call(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        return x

class SeparableTransposeConv2D(tf.keras.layers.Layer):
    """ Separable transpose convolutional layer"""
    def __init__(
            self,
            filters,
            kernel_size,
            strides=(1, 1),
            num_channels_in=1,
            activation=None,
            **kwargs):
        super(SeparableTransposeConv2D, self).__init__(**kwargs)
        kernel_size_time = kernel_size[0]
        self.filters = filters

        self.strides = strides
        self.activation = activation

        self.upsampling = tf.keras.layers.UpSampling2D(size=(1, 2))
        self.padding = tf.keras.layers.ZeroPadding2D(
                padding=((kernel_size_time-1, kernel_size_time-1),(0,kernel_size_time))
            )
        self.deconv = SeparableConv2D(
                self.filters,
                kernel_size,
                num_channels_in=num_channels_in,
                activation=activation)

    def call(self, inputs):
        """ Forward pass"""
        x = inputs
        x = self.upsampling(x)

        x = self.padding(x)
        x = self.deconv(x)

        return x

class encoder_unet(tf.keras.layers.Layer):
    """ Encoder of UNet"""
    def __init__(
            self,
            output_size=32,
            batch_size=1,
            kernel_size_time=3,
            num_chs=[1, 2, 4, 8, 16],
            separable=False,
            activation='tanh',
            **kwargs):
        super(encoder_unet, self).__init__(**kwargs)
        self.units = output_size
        self.batch_size = batch_size
        self.convs=[]
        self.num_chs = num_chs
        self.kernel_size_time = kernel_size_time
        self.freq_bins = [257, 128, 63, 31, 15]
        stages = len(self.num_chs) - 1


        for i , num_ch, num_ch_in in zip(range(stages), self.num_chs[1:], self.num_chs[:-1]):

            layer=tf.keras.Sequential(name=f"encoder_{i}")

            if separable:
                layer.add(
                    SeparableConv2D(
                        filters=num_ch,
                        kernel_size=(self.kernel_size_time, 3),
                        strides=(1, 2),
                        activation=activation,
                        num_channels_in=num_ch_in,
                        name=f"conv_{i}"
                        ))
            else:
                layer.add(
                    tf.keras.layers.Conv2D(
                        filters=num_ch,
                        kernel_size=(self.kernel_size_time, 3),
                        strides=(1, 2),
                        padding='valid',
                        activation=activation,
                        kernel_initializer='he_normal',
                        name=f"conv_{i}"
                        ))

            self.convs += [layer]

    def make_states(self):
        """ Make states"""
        states = []
        for num_ch, freq_bin in zip(self.num_chs[:-1], self.freq_bins[:-1]):
            states += [tf.zeros((self.batch_size, self.kernel_size_time-1, freq_bin, num_ch))]
        return states

    def call(
            self,
            inputs,
            states,
            training=False):
        """ Forward pass"""

        x = inputs
        outputs= []
        states_udpate=[]
        for state, net in zip(states, self.convs):
            states_udpate += [tf.identity(x[:,-(self.kernel_size_time-1):,:,:])]
            x = tf.concat([state, x], axis=1)
            x = net(x)
            outputs+= [x]

        return outputs, states_udpate

class decoder_unet(tf.keras.layers.Layer):
    """ Decoder of UNet"""
    def __init__(
            self,
            output_size=32,
            batch_size=1,
            kernel_size_time=3,
            num_chs=[1, 2, 4, 8, 16],
            separable=False,
            is_causal=True,
            activation='tanh',
            **kwargs):
        super(decoder_unet,self).__init__(**kwargs)
        self.num_chs = num_chs
        self.units = output_size
        self.batch_size = batch_size
        self.kernel_size_time = kernel_size_time
        self.convs=[]
        stages = len(self.num_chs) - 1
        # input shape (batch, T, Freq, 1)
        self.pad_freq_bins = [0,1, 0, 0]
        for i, num_ch, num_ch_in in zip(range(stages), self.num_chs[:-1], self.num_chs[1:]):
            num_pad = self.pad_freq_bins[i]
            layer=tf.keras.Sequential(name=f"decoder_{i}")
            if separable:
                layer.add(
                    SeparableTransposeConv2D(
                        filters=num_ch,
                        kernel_size=(kernel_size_time, 3),
                        activation=activation,
                        num_channels_in=num_ch_in,
                        name=f"conv_tran_{i}"
                        ))
            else:
                layer.add(
                    tf.keras.layers.Conv2DTranspose(
                        filters=num_ch,
                        kernel_size=(kernel_size_time, 3),
                        strides=(1, 2),
                        padding='valid',
                        activation=activation,
                        kernel_initializer='he_normal',
                        name=f"conv_tran_{i}"
                        ))
            layer.add(
                tf.keras.layers.ZeroPadding2D(
                    padding=((0, 0),(0,num_pad)),
                    name=f"padding_{i}")
            )

            layer.add(
                SliceLayer(
                    kernel_size_time,
                    is_causal=is_causal,
                    name=f"slice_{i}"))
            self.convs = [layer] + self.convs # reverse the order


    def call(self, inputs, training=False):
        """ Forward pass"""

        x = tf.identity(inputs[-1])
        for da, net in zip(inputs[::-1], self.convs):

            da1 = tf.concat([da, x], axis=-1)

            x = net(da1)

        return x

class unet(tf.keras.layers.Layer):
    """ UNet"""
    def __init__(
            self,
            output_size=32,
            batch_size=8,
            kernel_size_time=3,
            # num_chs = [1, 8, 16, 32, 64],
            num_chs=[1, 2, 4, 8, 16],
            separable=False,
            is_causal=True,
            activation='relu',
            **kwargs):
        super(unet,self).__init__(**kwargs)

        self.num_chs = num_chs
        self.units = output_size
        self.batch_size = batch_size
        self.encoder = encoder_unet(
            output_size=output_size,
            batch_size=batch_size,
            kernel_size_time=kernel_size_time,
            num_chs=self.num_chs,
            separable=separable,
            activation=activation)
        self.decoder = decoder_unet(
            output_size=output_size,
            batch_size=batch_size,
            kernel_size_time=kernel_size_time,
            num_chs=self.num_chs,
            separable=separable,
            is_causal=is_causal,
            activation= activation)
        self.batch_size=batch_size
        self.F = 15
        self.chs=self.num_chs[-1]
        self.rnn = tf.keras.layers.LSTM(self.F * self.chs, stateful=True, return_sequences=True)

    def call(self, inputs, states=None, training=False):
        """ Forward pass"""
        x = inputs

        outputs, states = self.encoder(x, states)
        T = tf.shape(outputs[-1])[1]
        out = tf.reshape(outputs[-1], (self.batch_size, T, -1))
        out = self.rnn(out)

        outputs[-1] = tf.reshape(
            out,
            (self.batch_size, T, self.F, self.chs))
        output = self.decoder(outputs)
        return output, states

if __name__ == "__main__":
    T = 500
    F = 257
    inputs = tf.random.uniform((8, T, F, 1))
    unet = unet(separable=True)
    states = unet.encoder.make_states()
    outputs,states = unet(inputs, states)
    print(outputs.shape)


    # encoder = encoder_unet()

    # outputs = encoder(inputs)

    # print("Encoder output shapes:")
    # for _ in outputs:
    #     print(_.shape)

    # decoder = decoder_unet()
    # outputs = decoder(outputs)
    # print("decoder output shapes:")
    # for _ in outputs:
    #     print(_.shape)
