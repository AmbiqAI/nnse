"""
UNet model in the streaming mode
see https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/7/125945/files/2024/04/Tan-Wang.taslp20-b86daf0e06ac52f1.pdf
"""
import tensorflow as tf
from .TransposeConv2D import TransposeConv2D, SeparableTransposeConv2D
from .dynamnic_tanh import dynamnic_tanh
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
            normalization_layer=None,
            **kwargs):
        super(SeparableConv2D, self).__init__(**kwargs)

        if normalization_layer is None:
            use_bias=True
        else:
            use_bias=False

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
            activation=activation,
            )

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
            time_steps=1,
            kernel_size_time=3,
            num_chs=[1, 2, 4, 8, 16],
            separable=False,
            activation='tanh',
            norm_mean=None,
            norm_inv_std=None,
            dim_feat=257,
            normalization_layer=None,
            **kwargs):
        super(encoder_unet, self).__init__(**kwargs)
        self.norm_mean = norm_mean
        self.norm_inv_std = norm_inv_std
        self.units = output_size
        self.batch_size = batch_size
        self.convs=[]
        self.num_chs = num_chs
        self.kernel_size_time = kernel_size_time

        self.freq_bins,_ = get_unet_info(
            num_chs,
            dim_feat=dim_feat)
        # self.freq_bins = [257, 128, 63, 31, 15]
        stages = len(self.num_chs) - 1
        self.states = self.make_states()
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
                        normalization_layer=normalization_layer,
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
            self):
        """ Make states"""
        states = []
        if self.norm_mean is not None:
            shape = (1, 1, -1, 1)
            norm_mean = tf.reshape(self.norm_mean, shape)
            norm_inv_std = tf.reshape(self.norm_inv_std, shape)
        else:
            norm_mean = self.norm_mean
            norm_inv_std = self.norm_inv_std
        len_pad = self.kernel_size_time - 1

        for i, da in enumerate(zip(self.num_chs[:-1], self.freq_bins[:-1])):
            num_ch, freq_bin = da
            shape = (self.batch_size, len_pad, freq_bin, num_ch)
            if i==0:
                state = tf.fill(shape, tf.math.log(2**-15) / tf.math.log(10.0))
                if norm_mean is not None:
                    state = (state - norm_mean) * norm_inv_std
                state = tf.Variable(state, trainable=False) # for eager mode
                states += [state]
            else:
                state=tf.zeros(shape)
                state = tf.Variable(state, trainable=False) # for eager mode
                states += [state]

        return states

    def reset_states(
            self):
        """ Reset states"""

        if self.norm_mean is not None:
            shape = (1, 1, -1, 1)
            norm_mean = tf.reshape(self.norm_mean, shape)
            norm_inv_std = tf.reshape(self.norm_inv_std, shape)
        else:
            norm_mean = self.norm_mean
            norm_inv_std = self.norm_inv_std

        len_pad = self.kernel_size_time - 1

        for i, da in enumerate(zip(self.num_chs[:-1], self.freq_bins[:-1])):
            num_ch, freq_bin = da
            shape = (self.batch_size, len_pad, freq_bin, num_ch)
            if i==0:
                state = tf.fill(shape, tf.math.log(2**-15) / tf.math.log(10.0))

                if norm_mean is not None:
                    state = (state - norm_mean) * norm_inv_std
                state = tf.Variable(state, trainable=False) # for eager mode
                self.states[i].assign(state)
            else:
                state=tf.zeros(shape)
                state = tf.Variable(state, trainable=False) # for eager mode
                self.states[i].assign(state)
    def call(
            self,
            inputs,
            # states,
            training=False):
        """ Forward pass"""

        x = inputs
        outputs= []

        for i, layer_info in enumerate(zip(self.states, self.convs)):
            state, net = layer_info
            x = tf.concat([state, x], axis=1)
            state_update=tf.identity(x[:,-(self.kernel_size_time-1):,:,:])
            x = net(x)
            self.states[i].assign(state_update)
            outputs+= [x]

        # self.states = states_udpate
        return outputs

class decoder_unet(tf.keras.layers.Layer):
    """ Decoder of UNet"""
    def __init__(
            self,
            output_size=32,
            batch_size=1,
            kernel_size_time=3,
            num_chs=[1, 2, 4, 8, 16],
            separable=False,
            activation='relu',
            dim_feat=257,
            time_steps=1,
            normalization_layer=None,
            **kwargs):
        super(decoder_unet,self).__init__(**kwargs)
        self.num_chs = num_chs
        self.units = output_size
        self.batch_size = batch_size
        self.kernel_size_time = kernel_size_time
        # (bins - kernel_size_time) // 2 + 1
        self.freq_bins, self.pad_freq_bins =  get_unet_info(
            num_chs,
            dim_feat=dim_feat)
        # self.freq_bins = [257, 128, 63, 31, 15]
        self.convs=[]
        stages = len(self.num_chs) - 1
        self.states=self.make_states()
        # input shape (batch, T, Freq, 1)
        # self.pad_freq_bins = [0,1, 0, 0]
        self.zeros = []

        for i, num_ch, num_ch_in in zip(range(stages), self.num_chs[:-1], self.num_chs[1:]):
            num_pad = self.pad_freq_bins[i]
            layer=tf.keras.Sequential(name=f"decoder_{i}")
            if i == 0:
               activation_layer=activation
            else:
                activation_layer = activation
            if separable:
                layer.add(
                    SeparableTransposeConv2D(
                        filters=num_ch,
                        kernel_size=(kernel_size_time, 3),
                        activation=activation_layer,
                        num_channels_in=num_ch_in*2,  # x2 since channel is doubled after concatenation
                        batch_size=batch_size,
                        time_steps=time_steps,
                        normalization_layer=normalization_layer,
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
            if num_pad > 0:
                zeros=tf.zeros((self.batch_size, time_steps, num_pad, num_ch), dtype=tf.float32)
                zeros = tf.Variable(zeros, trainable=False)
            else:
                zeros = None
  
            # layer.add(
            #     tf.keras.layers.ZeroPadding2D(
            #         padding=((0, 0),(0,num_pad)),
            #         name=f"padding_{i}")
            # )
            self.zeros = [zeros] + self.zeros # reverse the order
            self.convs = [layer] + self.convs # reverse the order

    def call(
            self,
            x,
            inputs_dec,
            # states=None,
            training=False):
        """ Forward pass"""

        for i, layer_info in enumerate(zip(inputs_dec[::-1], self.convs, self.states, self.zeros)):

            encode, net, state, zeros = layer_info
            state_en, state_de = state
            if self.kernel_size_time > 1:
                encode = tf.concat([state_en, encode], axis=1) # time concatenation
                x = tf.concat([state_de, x], axis=1) # time concatenation

                state_en_update=tf.identity(encode[:,-(self.kernel_size_time-1):,:,:])
                state_de_update=tf.identity(x[:,-(self.kernel_size_time-1):,:,:])
         
            comb = tf.concat([encode, x], axis=-1) # skip connection (channel concatenation)
            x = net(comb)
            # compensate the downsampling
            x = tf.concat([x, zeros], axis=2) if zeros is not None else x
            if self.kernel_size_time > 1:
                self.states[i][0].assign(state_en_update)
                self.states[i][1].assign(state_de_update)
        return x

    def make_states(self):
        """ Make states"""
        len_pad = self.kernel_size_time - 1

        states=[]
        for num_ch, freq_bin in zip(self.num_chs[1:], self.freq_bins[1:]):
            # reverse the order
            state_en = tf.zeros((self.batch_size, len_pad, freq_bin, num_ch))
            state_en = tf.Variable(state_en, trainable=False) # for eager mode
            state_de = tf.zeros((self.batch_size, len_pad, freq_bin, num_ch))
            state_de = tf.Variable(state_de, trainable=False) # for eager mode

            states = [(state_en, state_de)] + states

        return states

    def reset_state(self):
        """ Reset states"""
        for state in self.states:
            state[0].assign(state[0] * 0)
            state[1].assign(state[1] * 0)

class unet(tf.keras.layers.Layer):
    """ UNet"""
    def __init__(
            self,
            output_size=32,
            batch_size=8,
            time_steps=1,
            kernel_size_time=3,
            # num_chs = [1, 8, 16, 32, 64],
            num_chs=[1, 2, 4, 8, 16],
            separable=False,
            activation='relu',
            unroll_rnn=False,
            norm_mean=None,
            norm_inv_std=None,
            dim_feat=257,
            normalization_layer=None,
            **kwargs):
        super(unet,self).__init__(**kwargs)

        self.num_chs = num_chs
        self.units = output_size
        self.batch_size = batch_size
        self.kernel_size_time = kernel_size_time
        self.encoder = encoder_unet(
            output_size=output_size,
            batch_size=batch_size,
            time_steps=time_steps,
            kernel_size_time=kernel_size_time,
            num_chs=self.num_chs,
            separable=separable,
            activation=activation,
            norm_mean=norm_mean,
            norm_inv_std=norm_inv_std,
            dim_feat=dim_feat,
            normalization_layer=normalization_layer,)
        self.decoder = decoder_unet(
            output_size=output_size,
            batch_size=batch_size,
            time_steps=time_steps,
            kernel_size_time=kernel_size_time,
            num_chs=self.num_chs,
            separable=separable,
            activation= activation,
            dim_feat=dim_feat,
            normalization_layer=normalization_layer,)
        self.batch_size=batch_size

        self.freq_bins, self.pad_freq_bins = get_unet_info(
            num_chs,
            dim_feat=dim_feat)
        self.F=self.freq_bins[-1]
        self.chs=self.num_chs[-1]
        self.states = self.make_states()
        self.rnn = tf.keras.layers.LSTM(
            self.F * self.chs,
            return_state=True,
            stateful=False,
            unroll=unroll_rnn,
            return_sequences=True)

    def reset_states(self, zero_state=False):
        """ Reset states"""
        h_states = tf.Variable(
                            tf.random.uniform(
                                [self.batch_size, self.F * self.chs],
                                minval=-1,
                                maxval=1),
                            dtype = tf.float32,
                            trainable = False)

        c_states = tf.Variable(
                            tf.random.truncated_normal(
                                [self.batch_size, self.F * self.chs],
                                mean=0.0,
                                stddev=tf.sqrt(1 / self.F * self.chs)),
                            dtype = tf.float32,
                            trainable = False)
        if zero_state:
            h_states.assign(h_states * 0)
            c_states.assign(c_states * 0)
        self.states[0].assign(h_states)
        self.states[1].assign(c_states)
        self.encoder.reset_states()
        self.decoder.reset_state()

    def make_states(self, zero_state=False):
        """ Make states"""

        h_states = tf.Variable(
                            tf.random.uniform(
                                [self.batch_size, self.F * self.chs],
                                minval=-1,
                                maxval=1),
                            dtype = tf.float32,
                            trainable = False)

        c_states = tf.Variable(
                            tf.random.truncated_normal(
                                [self.batch_size, self.F * self.chs],
                                mean=0.0,
                                stddev=tf.sqrt(1 / self.F * self.chs)),
                            dtype = tf.float32,
                            trainable = False)
        if zero_state:
            h_states.assign(h_states * 0)
            c_states.assign(c_states * 0)

        return (h_states, c_states)

    def call(
            self,
            inputs,
            # states=None,
            training=False):
        """ Forward pass"""
        x = inputs
        # states_de = states

        # encoder
        outputs = self.encoder(x)

        # bottleneck rnn
        T = tf.shape(outputs[-1])[1]

        out = tf.reshape(
            outputs[-1],
            (self.batch_size, T, -1))
        out, h_state, c_state = self.rnn(out, initial_state=self.states)

        self.states[0].assign(h_state)
        self.states[1].assign(c_state)
        input_dec = tf.reshape(
            out,
            (self.batch_size, T, self.F, self.chs))
        
        # decoder
        output = self.decoder(
            input_dec,
            outputs)
        return output

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
