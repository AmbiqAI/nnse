"""
UNet model in the streaming mode
see https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/7/125945/files/2024/04/Tan-Wang.taslp20-b86daf0e06ac52f1.pdf
"""
import tensorflow as tf
from .TransposeConv2D import TransposeConv2D, SeparableTransposeConv2D

class SliceLayer(tf.keras.layers.Layer):
    """ Slice layer"""
    def __init__(
            self,
            kernel_size_time=3,
            **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.kernel_size_time = kernel_size_time

    def call(self, inputs):
        """ Forward pass"""
        outputs=inputs[:,(self.kernel_size_time-1):-(self.kernel_size_time-1),:,:] # remove the last (kernel_size_time-1) frames

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
        """ Forward pass"""
        x = self.depthwise(inputs)
        x = self.pointwise(x)
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
        self.freq_bins,_ = get_unet_info(num_chs)
        # self.freq_bins = [257, 128, 63, 31, 15]
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

    def make_states(
            self,
            norm_mean=None,
            norm_inv_std=None):
        """ Make states"""
        states = []
        if norm_mean is not None:
            shape = (1, 1, -1, 1)
            norm_mean = tf.reshape(norm_mean, shape)
            norm_inv_std = tf.reshape(norm_inv_std, shape)

        len_pad = self.kernel_size_time - 1

        for i, da in enumerate(zip(self.num_chs[:-1], self.freq_bins[:-1])):
            num_ch, freq_bin = da
            shape = (self.batch_size, len_pad, freq_bin, num_ch)
            if i==0:
                state = tf.fill(shape, tf.math.log(2**-15) / tf.math.log(10.0))

                if norm_mean is not None:
                    state = (state - norm_mean) * norm_inv_std

                states += [state]
            else:
                states += [tf.zeros(shape)]

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
            activation='tanh',
            **kwargs):
        super(decoder_unet,self).__init__(**kwargs)
        self.num_chs = num_chs
        self.units = output_size
        self.batch_size = batch_size
        self.kernel_size_time = kernel_size_time
        # (bins - kernel_size_time) // 2 + 1
        self.freq_bins, self.pad_freq_bins =  get_unet_info(num_chs)
        # self.freq_bins = [257, 128, 63, 31, 15]
        self.convs=[]
        stages = len(self.num_chs) - 1
        # input shape (batch, T, Freq, 1)
        # self.pad_freq_bins = [0,1, 0, 0]
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
                if 0:
                    layer.add(
                        TransposeConv2D(
                            filters=num_ch,
                            kernel_size=(kernel_size_time, 3),
                            activation=activation,
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
                        SliceLayer(
                            kernel_size_time,
                            name=f"slice_{i}"))

            layer.add(
                tf.keras.layers.ZeroPadding2D(
                    padding=((0, 0),(0,num_pad)),
                    name=f"padding_{i}")
            )

            self.convs = [layer] + self.convs # reverse the order


    def call(
            self,
            x,
            inputs_dec,
            states=None,
            training=False):
        """ Forward pass"""
        states_update=[]
        for encode, net, state in zip(inputs_dec[::-1], self.convs, states):
            states_update += [tf.identity(x[:,-(self.kernel_size_time-1):,:,:])]
            x = tf.concat([state, x], axis=1)
            encode = tf.pad( # padding encode on time axis
                encode,
                [[0, 0], [self.kernel_size_time-1, 0], [0, 0], [0, 0]])
            comb = tf.concat([encode, x], axis=-1) # skip connection

            x = net(comb)

        return x, states_update

    def make_states(self):
        """ Make states"""
        len_pad = self.kernel_size_time - 1

        states = []
        for num_ch, freq_bin in zip(self.num_chs[1:], self.freq_bins[1:]):
            # reverse the order
            states = [tf.zeros((self.batch_size, len_pad, freq_bin, num_ch))] + states
        return states

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
            activation='relu',
            unroll_rnn=False,
            **kwargs):
        super(unet,self).__init__(**kwargs)

        self.num_chs = num_chs
        self.units = output_size
        self.batch_size = batch_size
        self.kernel_size_time = kernel_size_time
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
            activation= activation)
        self.batch_size=batch_size
        self.F = 15
        self.chs=self.num_chs[-1]
        self.rnn = tf.keras.layers.LSTM(
            self.F * self.chs,
            stateful=True,
            unroll=unroll_rnn,
            return_sequences=True)

    def make_states(
            self,
            norm_mean=None,
            norm_inv_std=None):
        """ Make states"""
        states_en = self.encoder.make_states(norm_mean,norm_inv_std)
        states_de = self.decoder.make_states()
        return states_en, states_de

    def call(
            self,
            inputs,
            states=None,
            training=False):
        """ Forward pass"""

        x = inputs
        states_en, states_de = states

        # encoder
        outputs, states_en = self.encoder(x, states_en)

        # bottleneck rnn
        T = tf.shape(outputs[-1])[1]

        out = tf.reshape(
            outputs[-1],
            (self.batch_size, T, -1))
        out = self.rnn(out)

        input_dec = tf.reshape(
            out,
            (self.batch_size, T, self.F, self.chs))

        # decoder
        output, states_de = self.decoder(input_dec, outputs,states_de)
        return output, (states_en, states_de)

def get_unet_info(
        num_chs: list = [1, 2, 4, 8, 16],
        kernel_size_freq: int = 3,
        dim_feat: int = 257,
    ):
    """
    Print the information of the UNet:
    return:
        - freq_bins:
            Number of frequency bins in each stage. It includes
            the input (dim_feat)
            Formula: bins_update = (bins - kernel_size_freq) // 2 + 1
        - pad_freq_bins: 
            List of the number of padding frequency bins in each stage
    """
    stages = len(num_chs) - 1
    freq_bins = [dim_feat]
    num_bin = dim_feat
    pad_freq_bins = []
    for _ in range(stages):
        pad_freq_bins += [(num_bin - kernel_size_freq) % 2]
        num_bin = (num_bin - kernel_size_freq) // 2 + 1
        freq_bins += [num_bin]
    return freq_bins, pad_freq_bins

if __name__ == "__main__":
    T = 500
    F = 257
    inputs = tf.random.uniform((8, T, F, 1))
    unet = unet(separable=True)
    states = unet.make_states()
    outputs,states = unet(inputs, states)
    print(outputs.shape)
    get_unet_info()
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
