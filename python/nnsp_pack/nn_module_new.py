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

class NeuralNetClass(tf.keras.Model):
    """
    NeuralNetClass: a sequential model suppors only fc, lstm, 1D-conv layers
    """
    def __init__(
            self,
            config                 = None,
            batchsize              = 300):

        super(NeuralNetClass, self).__init__()
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
            kernel_initializer = self.weight_initializer(
                                        neuron_i,
                                        neuron_o,
                                        layer_type,
                                        activation)
            drop_rate = da['dropout']
            droprate_recurrent_layer = 0

            if layer_type == 'conv1d':
                kernel_size = da['kernel_size']
                self.kernel_size += [kernel_size]
                self.num_context = kernel_size[0]
                layer = layers.Conv2D(
                        neuron_o,
                        kernel_size,
                        padding     = 'valid',
                        strides     = (1, 1), # downsampling 2 in timesteps dim
                        activation  = activation,
                        kernel_initializer  = tf.keras.initializers.Constant(kernel_initializer),
                        input_shape = (None, neuron_i, 1)) # (time, dim_feat, ch)

            elif layer_type == 'conv2d':
                kernel_size = da['kernel_size']
                self.kernel_size += [kernel_size]
                layer = layers.Conv2D(
                        neuron_o,
                        kernel_size=kernel_size,
                        padding     = 'valid',
                        strides     = (1, 1), # downsampling 2 in timesteps dim
                        activation  = activation,
                        kernel_initializer  = tf.keras.initializers.Constant(kernel_initializer),
                        input_shape = (None, neuron_i, 1)) # (time, dim_feat, ch)

            elif layer_type == 'fc':
                layer = layers.Dense(
                        neuron_o,
                        activation = activation,
                        # kernel_initializer  = tf.keras.initializers.Constant(kernel_initializer)
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
                        unroll=False)
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
                    activation=activation)
                self.kernel_size_time = da['kernel_size_time']

            else:
                self.kernel_size += [None]
                drop_rate = 0 # already dropout in the lstm layer
            self.nn_layers += [layer]
            noise_shape=(None,1, None)
            self.dropout_layers+= [layers.Dropout(drop_rate, noise_shape = noise_shape)]


        self.stats_inst = statsClass(self.dim_trgt)
        self.weight_change = [
            tf.Variable(0, dtype = tf.float32, trainable = False)
                for i in range(20)]

        # run this to initialize the weight tables
        self.build_nn(batch_size=batchsize)

    def call(
        self,
        data_in,
        states,
        mask=1.0,
        training = False,
        quantized = False):

        """Calling function"""
        self.bitwidths['kernel'] = 8
        self.bitwidths['bias'] = 16

        # add the last dim to include channel (batches, timesteps, dim_feat, numCh = 1)
        out = self.input_layer(data_in)

        states_out=[]
        for i, layer_info in enumerate(zip(self.nn_layers, self.config[1:])):
            subnet, config = layer_info
            layer_type=config['layer_type']
            state = states[i]
            drop_layer = self.dropout_layers[i]
            out = drop_layer(out, training = training)

            if layer_type == 'conv1d':
                num_context = self.kernel_size[i][0]
                out = tf.concat([state, out], axis=-1)
                states_out += [out[:,-(num_context-1):,:]]
                out = tf.expand_dims(out,3)
                out = subnet(out, training = training) # (batches, timesteps, 1, neurons[1])
                out = out[:, :, 0, :]

            elif layer_type == 'conv2d':
                out = tf.expand_dims(out,3)
                out = subnet(out, training = training) # (batches, timesteps, dim_feat, num_filters)
                shape = tf.shape(out)
                out = tf.reshape(out, [shape[0], shape[1],-1])
                states_out += [state]

            elif layer_type == 'lstm':
                h_state, c_state = state
                out, h_state, c_state = subnet(
                                out,
                                initial_state = (h_state, c_state),
                                training = training)
                states_out += [(h_state, c_state)]

            elif layer_type == 'minGRU':
                out = subnet(out, return_states=False)
                states_out += [state]
            elif layer_type == 'unet':
                out = tf.expand_dims(out,-1)
                out, states_unet = subnet(out, state, training=training)
                out = out[:,:,:,0]
                states_out += [states_unet]
            else:
                out = subnet(out, training=training)
                states_out += [state]

        out *= mask
        self.update_limited_quantizated(quantized)

        return out, states_out

    def build_nn(self, quantized=False, batch_size=32):
        """
        Build your nn. This will provide the physical weight table.
        """
        timesteps=500
        states = self.make_states(batchsize = batch_size, zero_state = False)

        inputs = tf.constant(
            np.random.randn(
                batch_size,timesteps, self.config[0]['layer_neurons']),
                dtype = tf.float32)
        batch_size, _, dim_feat = inputs.shape

        masks = 1
        self.call(
            inputs,
            states,
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
                            tf.random.truncated_normal(
                                [batchsize, neuron_out],
                                stddev=1/np.sqrt(neuron_in)),
                            dtype = tf.float32,
                            trainable = False)
                # h_states.assign( tf.minimum(tf.maximum(h_states, -1.0), 1.0-2**-15) )
                c_states = tf.Variable(
                            tf.random.truncated_normal([batchsize, neuron_out]),
                            dtype = tf.float32,
                            trainable = False)
                if zero_state:
                    h_states.assign(h_states * 0)
                    c_states.assign(c_states * 0)
                states += [(h_states, c_states)]

            elif layer_type == 'conv1d':
                num_context = config_o['kernel_size'][0]
                shape = (batchsize, num_context-1, neuron_in)
                state = tf.fill(shape, tf.math.log(2**-15) / tf.math.log(10.0))
                if norm_mean is not None:
                    state = (state - norm_mean) * norm_inv_std
               
                states += [state]

            elif layer_type == 'unet':
                state = self.nn_layers[i].make_states(norm_mean,norm_inv_std)
                states += [state]

            else:
                states += [tf.zeros((batchsize, 1,1), dtype = tf.float32)]

        return states

    def quantized_weight(self):
        """
        Quantize the weight
        """
        self.build_nn(quantized=True)

    def weight_initializer(self, neuron_in, neuron_out, layerType, act_type):
        """
        Manually initialize weight table
        """
        if layerType == 'fc':
            shape = (neuron_in, neuron_out)
            tmp = np.random.normal(scale = 1.0 / np.sqrt(neuron_in), size = shape)
        elif layerType == 'conv1d':
            shape = (self.kernel_size, neuron_in, 1, neuron_out)
            tmp = np.random.normal(
                scale=1.0 / np.sqrt(shape[0] * shape[1] * shape[2]),
                size = shape)
        else:
            tmp = np.array([1])
        if re.search('relu', act_type):
            init_weight = tmp * np.sqrt(2)
        else:
            init_weight = tmp
        return init_weight.astype(np.float32)

    def duplicated_to(self, nn_duplx, logger):
        """
        Copy weight table from one net to the other.
        """
        nn_duplx.scalar_output = self.scalar_output
        nn_duplx.layer_types = self.layer_types.copy()
        nn_duplx.activaitons = self.activaitons.copy()
        nn_duplx.neurons = self.neurons.copy()
        names = [r'/bias', r'/kernel:0', r'/recurrent_kernel']
        k = 0
        eps = 10**-5

        for i, layer_pair in enumerate(zip(self.nn_layers, nn_duplx.nn_layers)):
            layer, layer_copy = layer_pair
            for val in layer.trainable_variables:
                for fd_name in names:
                    fd_src = re.search(fd_name, val.name)
                    if fd_src:
                        for val_fd in layer_copy.trainable_variables:
                            fd_dst = re.search(fd_name, val_fd.name)
                            if fd_dst:
                                den = tf.reduce_sum(tf.pow(val_fd - val,2.0))
                                num = tf.reduce_sum(tf.pow(val_fd,2.0)) + eps
                                self.weight_change[k].assign(tf.sqrt(den/num))
                                val_fd.assign(val)
                                k+=1

        string = '\n\tweight changing rate = '
        for i in range(k):
            string += '%3.5f ' % self.weight_change[i]
        string += '\n'
        logger.info(string)

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
