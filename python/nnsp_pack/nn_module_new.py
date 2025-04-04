"""
    NeuralNetClass: a sequential model supports only fc, lstm & 1D-conv layers
"""
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from .minRNN import minGRU
from .unet import unet
from . import post_aware_quant
from .statsClass import statsClass
# from tensorflow_model_optimization.python.core.keras.compat import keras
class NeuralNetClass(tf.keras.Model):
    """
    NeuralNetClass: a sequential model suppors only fc, lstm, 1D-conv layers
    """
    def __init__(
            self,
            config                 = None,
            batchsize              = 300,
            time_steps             = 1,
            unroll_rnn             = False,
            norm_mean              = None,
            norm_inv_std           = None,
            ):

        super(NeuralNetClass, self).__init__()
        self.time_steps = time_steps
        self.norm_mean = norm_mean
        self.norm_inv_std = norm_inv_std
        self.config = config
        self.dim_trgt = config[-1]['layer_neurons']
        self.dim_feat = config[0]['layer_neurons']

        self.num_layers = len(config)-1
        self.input_layer = tf.keras.layers.InputLayer(
                input_shape   = ((None, self.dim_feat)),
                batch_size    = batchsize)

        self.bitwidths = {  'kernel': 16,
                            'bias'  : 16 }
        self.nfracs = { 'kernel': [None] * self.num_layers,
                        'bias'  : [None] * self.num_layers }
        self.num_context = 1
        for i in range(self.num_layers):
            self.nfracs['kernel'][i] =  tf.Variable(12, dtype = tf.float32, trainable = False)
            self.nfracs['bias'][i] =  tf.Variable(12, dtype = tf.float32, trainable = False)

        self.nn_layers =[]
        self.dropout_layers = []
        self.kernel_size = []
        self.kernel_size_time = 1

        for da_former, da in zip(config[:-1], config[1:]):
            neuron_i = da_former['layer_neurons']
            neuron_o = da['layer_neurons']
            activation = da['activation']
            layer_type = da['layer_type']
            kernel_initializer = "he_normal" if activation in ('relu', 'relu6') else "glorot_uniform"

            drop_rate = da['dropout']
            droprate_recurrent_layer = 0

            if layer_type in ('conv1d', 'conv2d'):
                kernel_size = da['kernel_size'] if layer_type=='conv2d' else [da['kernel_size'], neuron_i]
                self.kernel_size += [kernel_size]
                self.num_context = kernel_size[0]
                layer = layers.Conv2D(
                        neuron_o,
                        kernel_size,
                        padding     = 'valid',
                        strides     = (1, 1),
                        activation  = activation,
                        kernel_initializer  = kernel_initializer,
                        input_shape = (None, neuron_i, 1)) # (time, dim_feat, ch)

            elif layer_type == 'fc':
                layer = layers.Dense(
                        neuron_o,
                        activation = activation,
                        # kernel_initializer = kernel_initializer
                        )
                self.kernel_size += [None]

            elif layer_type == 'lstm':
                layer = layers.LSTM(
                        neuron_o,
                        dropout = drop_rate,
                        recurrent_dropout = droprate_recurrent_layer,
                        return_sequences = True,
                        return_state = True,
                        stateful = False,
                        unit_forget_bias = True,
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        unroll=unroll_rnn)
                self.kernel_size += [None]

            elif layer_type == 'minGRU':
                layer = minGRU(
                        neuron_o)
                self.kernel_size += [None]

            elif layer_type == 'unet':
                layer= unet(
                    batch_size=batchsize,
                    separable=da['separable'],
                    num_chs = da['num_chs'],
                    kernel_size_time = da['kernel_size_time'],
                    activation=activation,
                    unroll_rnn=unroll_rnn,
                    norm_mean=norm_mean,
                    norm_inv_std=norm_inv_std,
                    time_steps=time_steps,
                    dim_feat=self.dim_feat,
                    normalization_layer=da['normalization_layer'],
                    )
                self.kernel_size_time = da['kernel_size_time']

            else:
                raise ValueError('Layer type not recognized')
                drop_rate = 0 # already dropout in the lstm layer
            self.nn_layers += [layer]
            noise_shape=(None,1, None)
            self.dropout_layers+= [layers.Dropout(drop_rate, noise_shape = noise_shape)]
        
        self.states= self.make_states(batchsize, norm_mean, norm_inv_std)
        self.stats_inst = statsClass(self.dim_trgt)
        self.weight_change = [
            tf.Variable(0, dtype = tf.float32, trainable = False)
                for i in range(20)]

        # run this to initialize the weight tables
        self.build_nn(batch_size=batchsize)

    def call(
        self,
        data_in,
        mask=1.0,
        training = False,
        quantized = False):

        """Calling function"""
        self.bitwidths['kernel'] = 8
        self.bitwidths['bias'] = 16

        # add the last dim to include channel (batches, timesteps, dim_feat, numCh = 1)
        out = self.input_layer(data_in)

        for i, layer_info in enumerate(zip(self.nn_layers, self.config[1:])):
            subnet, config = layer_info
            layer_type=config['layer_type']
            state = self.states[i]
            drop_layer = self.dropout_layers[i]
            out = drop_layer(out, training = training)

            if layer_type in ('conv1d', 'conv2d'):
                if layer_type=='conv1d':
                    num_context_time = config['kernel_size']
                else:
                    num_context_time = config['kernel_size'][0]

                out = tf.concat([state, out], axis=-2)
                state_update= tf.identity(out[:,-(num_context_time-1):,:])
                out = tf.expand_dims(out, 3)
                out = subnet(out, training = training) # (batches, timesteps, 1, neurons[1])
                out = out[:, :, 0, :]

                state.assign(state_update)

            elif layer_type == 'lstm':
                h_state, c_state = state
                out, h_state_update, c_state_update = subnet(
                                out,
                                initial_state = (h_state, c_state),
                                training = training)
                h_state.assign(h_state_update)
                c_state.assign(c_state_update)

            elif layer_type == 'minGRU':
                out = subnet(out, return_states=False)

            elif layer_type == 'unet':
                out = tf.expand_dims(out,-1)
                out = subnet(out, training=training)
                out = out[:,:,:,-1]
            else:
                out = subnet(out, training=training)

        out *= mask
        # self.update_limited_quantizated(quantized)

        return out

    def get_config_info(self, info='layer_type'):
        """
        Get the configuration information
        """
        if info == "layer_neurons":
            config_info = [self.config[0]['layer_neurons']]
        else:
            config_info = []
        for da in self.config[1:]:
            config_info.append(da[info])
        return config_info
            

    def build_nn(self, quantized=False, batch_size=32):
        """
        Build your nn. This will provide the physical weight table.
        """

        # states = self.make_states(batchsize = batch_size, zero_state = False)
        inputs = tf.constant(
            np.random.randn(
                batch_size, self.time_steps, self.config[0]['layer_neurons']),
                dtype = tf.float32)
        batch_size, _, dim_feat = inputs.shape
        
        masks = 1

        self.call(
            inputs,
            quantized=quantized)

    def make_states(
            self,
            batchsize,
            norm_mean=None,
            norm_inv_std=None,
            zero_state = False):
        """
        Initalize lstm states
        """
        config=self.config
        states = []
        for i, data in enumerate(zip(config[:-1], config[1:])):
            config_i, config_o = data
            neuron_in = config_i['layer_neurons']
            neuron_out = config_o['layer_neurons']
            layer_type = config_o['layer_type']
            if layer_type == 'lstm':
                h_states = tf.Variable(
                            tf.random.uniform(
                                [batchsize, neuron_out],
                                minval = -1.0,
                                maxval = 1.0),
                            dtype = tf.float32,
                            trainable = False)
                c_states = tf.Variable(
                            tf.random.truncated_normal([batchsize, neuron_out]),
                            dtype = tf.float32,
                            trainable = False)
                if zero_state:
                    h_states.assign(h_states * 0)
                    c_states.assign(c_states * 0)
                states += [(h_states, c_states)]

            elif layer_type in ('conv1d', 'conv2d'):
                num_context = config_o['kernel_size'] if layer_type=="conv1d" else config_o['kernel_size'][0] 
                shape = (batchsize, num_context-1, neuron_in)
                state = tf.fill(shape, tf.math.log(2**-15) / tf.math.log(10.0))
                state = tf.Variable( # for eager mode assign
                    state,
                    dtype = tf.float32,
                    trainable = False)
                if norm_mean is not None:
                    state.assign( (state - norm_mean) * norm_inv_std )

                states += [state]

            elif layer_type == 'unet':
                states += [ self.nn_layers[i].states]

            else:
                states += [None]

        return states

    def reset_states(self, zero_state):
        """
        Reset the states
        """
        for i, layer_info in enumerate(zip(self.nn_layers, self.config[1:])):
            subnet, config = layer_info
            layer_type = config['layer_type']
            if layer_type == 'lstm':
                h_states, c_states = self.states[i]
                h = tf.Variable(
                            tf.random.uniform(
                                h_states.shape,
                                minval = -1.0,
                                maxval = 1.0),
                            dtype = tf.float32,
                            trainable = False)
                c = tf.Variable(
                            tf.random.truncated_normal(
                                c_states.shape),
                            dtype = tf.float32,
                            trainable = False)
                if zero_state:
                    h_states.assign(h * 0)
                    c_states.assign(c * 0)
            elif layer_type in ('conv1d', 'conv2d'):
                shape = self.states[i].shape
                state = tf.Variable(
                    tf.fill(shape, tf.math.log(2**-15) / tf.math.log(10.0)),
                    dtype = tf.float32,
                    trainable = False)
                if self.norm_mean is not None:
                    state = state.assign( (state - self.norm_mean) * self.norm_inv_std)
                self.states[i].assign(state)
            elif layer_type == 'unet':
                subnet.reset_states(zero_state=zero_state)

    def quantized_weight(self, batch_size=1):
        """
        Quantize the weight
        """
        self.build_nn(quantized=True, batch_size=batch_size)

    def update_limited_quantizated(
            self,
            quantized,
            limited = True,
            bounds_bias = [-4, 4 - 2**-6],
            bounds_weight = [-8, 8 - 2**-5],
            bitwidth_weight = 8,
            bitwidth_bias = 16):
        """
        Limiter and quantization of weight tables
        """
        for i, layer_info in enumerate(zip(self.nn_layers, self.config[1:])):
            layer, config = layer_info
            qbits_w = self.nfracs['kernel'][i]
            qbits_b = self.nfracs['bias'][i]
            layer_type = config['layer_type']
            if layer_type == 'conv1d' or layer_type == 'fc':
                for val in layer.trainable_variables:
                    if re.search(r'bias', val.name):
                        bias = val
                    elif re.search(r'/kernel:0', val.name):
                        kernel = val
                if limited:
                    post_aware_quant.limiter(kernel, bounds_weight)
                    post_aware_quant.limiter(bias, bounds_bias)
                if quantized:
                    post_aware_quant.get_frac_bit(kernel, bitwidth_weight, qbits_w)
                    post_aware_quant.fake_quantization(kernel, bitwidth_weight, qbits_w)
                    post_aware_quant.get_frac_bit(bias, bitwidth_bias, qbits_b)
                    post_aware_quant.fake_quantization(bias, bitwidth_bias, qbits_b)

            elif layer_type == 'lstm':
                for val in layer.trainable_variables:
                    if re.search(r'bias', val.name):
                        bias = val
                    elif re.search(r'/kernel:0', val.name):
                        kernel = val
                    elif re.search(r'/recurrent_kernel', val.name):
                        recurrent_kernel = val

                if limited:
                    post_aware_quant.limiter(kernel, bounds_weight)
                    post_aware_quant.limiter(recurrent_kernel, bounds_weight)
                    post_aware_quant.limiter(bias, bounds_bias)

                if quantized:
                    qbits_f = post_aware_quant.get_frac_bit(kernel, bitwidth_weight, qbits_w)
                    qbits_r = post_aware_quant.get_frac_bit(
                                        recurrent_kernel,
                                        bitwidth_weight,
                                        qbits_w)
                    qbits_w.assign(tf.minimum(qbits_f, qbits_r))
                    post_aware_quant.fake_quantization(kernel, bitwidth_weight, qbits_w)
                    post_aware_quant.fake_quantization(
                                        recurrent_kernel,
                                        bitwidth_weight,
                                        qbits_w)

                    qbits_b = post_aware_quant.get_frac_bit(bias, bitwidth_bias, qbits_b)
                    post_aware_quant.fake_quantization(bias, bitwidth_bias, qbits_b)
