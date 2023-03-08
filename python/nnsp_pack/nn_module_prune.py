"""
    NeuralNetClass: a sequential model supports only fc, lstm & 1D-conv layers
"""
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from . import post_aware_quant

class NeuralNetClass(tf.keras.Model):
    """
    NeuralNetClass: a sequential model suppors only fc, lstm, 1D-conv layers
    """
    def __init__(self,
                 batchsize              = 300,
                 layer_types            = ['fc'] * 10,
                 activations            = ['tanh'] * 10,
                 nDownSample            = 2,
                 kernel_size            = 6,
                 neurons                = [10] * 10,
                 dropRates              = [0] * 10,
                 dropRates_recurrent    = [0] * 10,
                 dim_target = 7):

        super(NeuralNetClass, self).__init__()
        self.kernel_size = kernel_size
        self.dim_trgt = neurons[-1]
        self.dim_feat = neurons[0]
        self.num_layers = len(layer_types)
        self.nDownSample = nDownSample
        self.neurons = neurons.copy()
        self.layer_types = layer_types.copy()
        self.activaitons = activations.copy()
        self.input_layer = tf.keras.layers.InputLayer(
                input_shape   = ((None, neurons[0])),
                batch_size    = batchsize)

        self.nn_layers = [None] * self.num_layers
        self.dropout_layers = [None] * self.num_layers
        self.bitwidths = {  'kernel': 16,
                            'bias'  : 16 }
        self.nfracs = { 'kernel': [None] * self.num_layers,
                        'bias'  : [None] * self.num_layers }
        self.h_states = [None] * len(self.layer_types)
        self.c_states = [None] * len(self.layer_types)

        for i in range(self.num_layers):
            self.nfracs['kernel'][i] =  tf.Variable(12, dtype = tf.float32, trainable = False)
            self.nfracs['bias'][i] =  tf.Variable(12, dtype = tf.float32, trainable = False)

        for i, neuron in enumerate(neurons[1:]):

            layer_type = layer_types[i]
            kernel_initializer = self.weight_initializer(
                                        neurons[i],
                                        neuron,
                                        layer_type,
                                        activations[i])
            drop_rate = dropRates[i]
            droprate_recurrent_layer = dropRates_recurrent[i]

            if layer_type == 'conv1d':
                layer = Conv2DPruneLayer(
                        neuron,
                        (kernel_size, neurons[i]),
                        padding     = 'valid',
                        strides     = (self.nDownSample, 1), # downsampling 2 in timesteps dim
                        activation  = activations[i],
                        kernel_initializer  = tf.keras.initializers.Constant(kernel_initializer),
                        input_shape = (None, neurons[i], 1)) # (time, dim_feat, ch)

            elif layer_type == 'fc':
                if i == self.num_layers-1:
                    layer = layers.Dense(
                            neuron,
                            activation = activations[i],
                            kernel_initializer  = tf.keras.initializers.Constant(kernel_initializer)) # pylint: disable=line-too-long
                else:
                    layer = DensePruneLayer(
                            neuron,
                            activation = activations[i],
                            kernel_initializer  = tf.keras.initializers.Constant(kernel_initializer)) # pylint: disable=line-too-long

            elif layer_type == 'lstm':
                layer = LSTMPruneLayer(
                        neuron,
                        dropout = drop_rate,
                        recurrent_dropout = droprate_recurrent_layer,
                        return_sequences = True,
                        return_state = True,
                        stateful = False,
                        unit_forget_bias = True,
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        unroll=False)

                drop_rate = 0 # already dropout in the lstm layer
            self.nn_layers[i] = layer
            noise_shape=(None,1, None)
            self.dropout_layers[i] = layers.Dropout(drop_rate, noise_shape = noise_shape)
        self.stats = {'acc_loss'        : tf.Variable(0, trainable = False, dtype = tf.float64),
                      'acc_steps'       : tf.Variable(0, trainable = False, dtype = tf.float64),
                      'acc_matchCount'  : tf.Variable(0, trainable = False, dtype = tf.float64)
                      }

        self.confusion_mat = tf.zeros((dim_target , dim_target), dtype=tf.float64)

        self.weight_change = [
            tf.Variable(0, dtype = tf.float32, trainable = False)
                for i in range(20)]

        # run this to initialize the weight tables
        self.build_nn()

    def call(
        self,
        data_in,
        mask,
        states,
        training = False,
        quantized = False,
        hard_thresh = None):

        """Calling function"""
        self.bitwidths['kernel'] = 8
        self.bitwidths['bias'] = 16
        (h_states, c_states) = states
        # add the last dim to include channel (batches, timesteps, dim_feat, numCh = 1)
        out = self.input_layer(data_in)
        reg_lst = []

        for i, layer in enumerate(self.nn_layers):
            drop_layer = self.dropout_layers[i]
            out = drop_layer(out, training = training)
            if self.layer_types[i] == 'conv1d':
                out = tf.expand_dims(out,3)
                # out: (batches, timesteps, 1, neurons[1])
                out, reg_layer, prune_mask = layer(out, training = training)
                out = out[:, :, 0, :]
                if hard_thresh:
                    if not training:
                        prune_mask = tf.cast(reg_layer >= hard_thresh, tf.float32)
                else:
                    prune_mask = 1
                reg_lst += [tf.reduce_sum(reg_layer * prune_mask)]
                out = out * prune_mask

            elif self.layer_types[i] == 'lstm':
                out, h_state, c_state, reg_layer, prune_mask = layer(
                                out,
                                initial_state = (h_states[i], c_states[i]),
                                training = training)
                if hard_thresh:
                    if not training:
                        prune_mask = tf.cast(reg_layer >= hard_thresh, tf.float32)
                else:
                    prune_mask = 1
                reg_lst += [tf.reduce_sum(reg_layer * prune_mask)]
                out = out * prune_mask
                h_state = h_state * prune_mask
                c_state = c_state * prune_mask

                self.h_states[i] = h_state
                self.c_states[i] = c_state
            else:
                if i==self.num_layers-1:
                    out = layer(out, training=training)
                else:
                    out, reg_layer, prune_mask = layer(out, training=training)
                    if hard_thresh:
                        if not training:
                            prune_mask = tf.cast(reg_layer >= hard_thresh, tf.float32)
                    else:
                        prune_mask = 1
                    reg_lst += [tf.reduce_sum(reg_layer * prune_mask)]
                    out = out * prune_mask

        states = (self.h_states, self.c_states)
        out *= mask
        self.update_limited_quantizated(quantized)
        reg = tf.reduce_sum(reg_lst)
        return out, states, reg

    def build_nn(self, quantized=False):
        """
        Build your nn. This will provide the physical weight table.
        """
        states = lstm_states(self, batchsize = 1, zero_state = False)
        inputs = tf.constant(np.random.randn(1,1, self.neurons[0]), dtype = tf.float32)
        batch_size, _, dim_feat = inputs.shape
        zero_pad = tf.zeros((batch_size, self.kernel_size -1, dim_feat), dtype = tf.float32)
        inputs_pad = tf.concat([zero_pad, inputs], 1)
        masks = 1
        self.call(inputs_pad, masks, states, quantized=quantized)
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
        nn_duplx.layer_types = self.layer_types.copy()
        nn_duplx.activaitons = self.activaitons.copy()
        nn_duplx.neurons = self.neurons.copy()
        names = [r'/bias', r'/kernel:0', r'/recurrent_kernel', r'/prune_threshold']
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

        string = "\n\tweight changing rate = "
        for i in range(k):
            string += f"{self.weight_change[i]} "
        string += "\n"
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
        for i, layer in enumerate(self.nn_layers):
            qbits_w = self.nfracs['kernel'][i]
            qbits_b = self.nfracs['bias'][i]

            if self.layer_types[i] == 'conv1d' or self.layer_types[i] == 'fc':
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

            elif self.layer_types[i] == 'lstm':
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

    def update_cost_steps(self, ave_loss, steps):
        """
        update total cost and total steps
        """
        stats = self.stats
        ave_loss = tf.cast(ave_loss, tf.float64)
        steps = tf.cast(steps, tf.float64)
        stats['acc_loss'].assign(stats['acc_loss'] + ave_loss * steps)
        stats['acc_steps'].assign(stats['acc_steps'] + steps)

    def update_accuracy(self, trgt0, est0, mask, dim):
        """
        Update total correct number of estimation
        """
        tmp = tf.reduce_sum(
                tf.cast(tf.math.equal(est0, trgt0), tf.float64)
                 * tf.cast(mask[:,:,0], tf.float64))
        self.stats['acc_matchCount'].assign_add(tf.cast(tmp, tf.float64))

        mat = [None] * dim * dim
        mask = tf.cast(mask, tf.int64)[:,:,0]
        for i in range(dim):
            est1 = est0 * mask
            trgt1 = trgt0 * mask
            mask_i = tf.cast(tf.math.equal(trgt1, i), dtype = tf.int64) * mask
            for j in range(dim):
                tmp = tf.cast(tf.math.equal(est1, j), dtype = tf.int64) * mask_i
                mat[i * dim + j] = tf.reduce_sum(tmp)
        mat = tf.convert_to_tensor(mat)
        mat = tf.reshape(mat, (dim, dim))

        self.confusion_mat = self.confusion_mat + tf.cast(mat, tf.float64)

    def show_loss(self, loss, total_accuracy, steps, show_step=True):
        """
        Print the loss function
        """
        loss  = tf.cast(loss,  tf.float64)
        steps = tf.cast(steps, tf.float64)
        loss0 = tf_round( loss / steps / self.neurons[-1] )
        accuracy0 = tf_round( total_accuracy / steps )

        tf.print('loss: ', loss0, end = '')
        tf.print(', accuracy: ', accuracy0, end = '')

        if show_step:
            tf.print( ' (', steps, ')', end = '')

    def show_confusion_matrix(self, dim_intent, logger):
        """
        Print the confusion matrix
        """
        mat = self.confusion_mat
        string = ''
        string += 'confused Matrix \n'
        for i in range(dim_intent):
            tmp = 0
            # tf.print('\tclass %d: ' % i, end = '')
            for j in range(dim_intent):
                tmp = tmp + mat[i][j]
            for j in range(dim_intent):
                mat_ij = tf_round(mat[i][j] / tmp)
                string += f'{mat_ij:2.2f} '
            string+='\n'
        logger.info(string)

    def reset_stats(self):
        """
        Reset the internal states of the neural net
        """
        for _, value in self.stats.items():
            value.assign(0)

        self.confusion_mat = self.confusion_mat * 0

def tf_round(data_in, prec = 10):
    """
    Round the tensor
    """
    base = tf.pow(tf.constant(10.0, dtype = data_in.dtype), prec)
    return tf.round(data_in * base) / base

def lstm_states(net, batchsize, zero_state = False):
    """
    Initalize lstm states
    """
    h_states = [None] * len(net.layer_types)
    c_states = [None] * len(net.layer_types)
    for i, nn_t in enumerate(net.layer_types):
        if nn_t == 'lstm':
            h_states[i] = tf.Variable(
                        tf.random.truncated_normal(
                            [batchsize, net.neurons[i+1]],
                            stddev=1/np.sqrt(net.neurons[i])),
                        dtype = tf.float32,
                        trainable = False)
            h_states[i].assign( tf.minimum(tf.maximum(h_states[i], -1.0), 1.0-2**-15) )
            c_states[i] = tf.Variable(
                        tf.random.truncated_normal([batchsize, net.neurons[i+1]]),
                        dtype = tf.float32,
                        trainable = False)
            if zero_state:
                h_states[i].assign(h_states[i] * 0)
                c_states[i].assign(c_states[i] * 0)

    return (h_states, c_states)
#%% Build the model

def ssigmoid(inputs):
    """
    scale sigmoid
    """
    return tf.math.sigmoid(10 * inputs)

class DensePruneLayer(tf.keras.layers.Dense):
    """
    Dense prune layer
    """
    def build(self, input_shape):
        """
        build
        """
        super().build(input_shape)

        self.prune_threshold = self.add_weight( # pylint: disable=attribute-defined-outside-init
            "prune_threshold",
            shape=[],
            initializer=tf.keras.initializers.Constant([0.5,]),
            dtype=self.kernel.dtype,
            trainable=False,
            aggregation=tf.VariableAggregation.NONE,
        )
        reg = self.get_prune_reg()
        sort_reg = tf.sort(reg, direction='ASCENDING')
        # self.prune_threshold.assign(sort_reg[sort_reg.shape[0] >> 2])

    def call(self, inputs):
        """
        call
        """
        outputs = super().call(inputs)

        reg, prune_mask = self.get_prune_reg_mask()

        return outputs, reg, prune_mask

    def get_prune_reg(self):
        """
        get_prune_reg
        """
        l2_norm_square = tf.reduce_sum((self.kernel)**2, axis=0)
        l2_norm_square += self.bias**2
        norm_gl2 = tf.math.sqrt(l2_norm_square + 2**-15)
        return norm_gl2

    def get_prune_mask(self, reg):
        """
        get_prune_mask
        """
        prune_mask = ssigmoid(reg - tf.math.sqrt(self.prune_threshold**2 + 2**-15))

        return prune_mask

    def get_prune_reg_mask(self):
        """
        get_prune_mask_reg
        """
        reg = self.get_prune_reg()
        prune_mask = self.get_prune_mask(reg)

        return reg, prune_mask

class Conv2DPruneLayer(tf.keras.layers.Conv2D):
    """
    Conv2 prune layer
    """
    def build(self, input_shape):
        """
        build
        """
        super().build(input_shape)

        self.prune_threshold = self.add_weight( # pylint: disable=attribute-defined-outside-init
            "prune_threshold",
            shape=[],
            initializer=tf.keras.initializers.Constant([0.5,]),
            dtype=self.kernel.dtype,
            trainable=False,
            aggregation=tf.VariableAggregation.NONE,
        )
        reg = self.get_prune_reg()
        sort_reg  = tf.sort(reg,direction='ASCENDING')
        # self.prune_threshold.assign(sort_reg[sort_reg.shape[0] >> 2])

    def call(self, inputs):
        """
        call
        """
        outputs = super().call(inputs)

        reg, prune_mask = self.get_prune_reg_mask()

        return outputs, reg, prune_mask

    def get_prune_reg(self):
        """
        get_prune_reg
        """
        l2_norm_square = tf.reduce_sum((self.kernel)**2, axis=[0,1,2])
        l2_norm_square += self.bias**2
        norm_gl2 = tf.math.sqrt(l2_norm_square +2**-15)
        return norm_gl2

    def get_prune_mask(self, reg):
        """
        get_prune_mask
        """
        prune_mask = ssigmoid(reg - tf.math.sqrt(self.prune_threshold**2 + 2**-15))

        return prune_mask

    def get_prune_reg_mask(self):
        """
        get_prune_mask_reg
        """
        reg = self.get_prune_reg()
        prune_mask = self.get_prune_mask(reg)

        return reg, prune_mask

class LSTMPruneLayer(tf.keras.layers.LSTM):
    """
    my LSTM prune layer
    """
    def build(self, input_shape):
        """
        build
        """
        super().build(input_shape)

        self.prune_threshold = self.add_weight( # pylint: disable=attribute-defined-outside-init
            "prune_threshold",
            shape=[],
            initializer=tf.keras.initializers.Constant([0.5,]),
            dtype=self.trainable_variables[0].dtype,
            trainable=False,
            aggregation=tf.VariableAggregation.NONE,
        )
        reg= self.get_prune_reg()
        sort_reg  = tf.sort(reg,direction='ASCENDING')
        # self.prune_threshold.assign(sort_reg[sort_reg.shape[0] >> 2])

    def call(
            self,
            inputs,
            mask=None,
            training=None,
            initial_state=None):
        """
        call
        """
        outputs, h_state, c_state = super().call(inputs, mask, training, initial_state)

        reg, prune_mask = self.get_prune_reg_mask()

        return outputs, h_state, c_state, reg, prune_mask

    def get_prune_reg(self):
        """
        get_prune_reg
        """
        # group reg for weight
        weights = [ self.trainable_variables[0]**2,
                    self.trainable_variables[1]**2]
        l2_norm_square_weight = tf.concat(weights, 0)
        l2_norm_square_weight = tf.split(l2_norm_square_weight, 4, -1)
        l2_norm_square_weight = tf.concat(l2_norm_square_weight, 0)
        # group reg for bias
        l2_norm_square_bias = self.trainable_variables[2]**2
        l2_norm_square_bias = tf.reshape(l2_norm_square_bias, (4,-1))
        # total group reg
        weight_bias = [l2_norm_square_weight, l2_norm_square_bias]
        l2_norm_square_weight = tf.concat(weight_bias, 0)
        l2_norm_square = tf.reduce_sum(l2_norm_square_weight, axis=0)

        norm_gl2 = tf.math.sqrt(l2_norm_square + 2**-15)
        return norm_gl2

    def get_prune_mask(self, reg):
        """
        get_prune_mask
        """
        prune_mask = ssigmoid(reg - tf.math.sqrt(self.prune_threshold**2 + 2**-15))

        return prune_mask

    def get_prune_reg_mask(self):
        """
        get_prune_mask_reg
        """
        reg = self.get_prune_reg()
        prune_mask = self.get_prune_mask(reg)

        return reg, prune_mask
