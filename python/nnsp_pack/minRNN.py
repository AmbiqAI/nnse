""" Minimal RNN
    This is a minimal implementation of the Gated Recurrent Unit (GRU) cell and
    a minimal Long Short-Term Memory (LSTM) cell
    using `tf.scan` function.
    See https://arxiv.org/pdf/2410.01201?
"""
import tensorflow as tf
def scan(u, v, initializer):
    """ Scan function for parallel computation
        a[batch, t, :] = u[batch, t, :] ⊙ a[batch, t-1, :] + v[batch, t, :].
        Inputs:
            u: a tensor of shape (batchsize, time_step, dim_out)
            v: a tensor of shape (batchsize, time_step, dim_out)
            initializer: a tensor of shape (batchsize, dim_out)
        Returns:
            outputs: a tensor of shape (batchsize, time_step, dim_out)
            outputs_t[-1]: a tensor of shape (batchsize, dim_out)
    """
    vt = tf.transpose(v, perm=[1, 0, 2])
    ut = tf.transpose(u, perm=[1, 0, 2])
    outputs_t = tf.scan(
        lambda a, x: x[0] * a + x[1],
        (ut, vt),
        initializer,
        parallel_iterations=100,
        swap_memory=True,
        )
    outputs = tf.transpose(outputs_t, perm=[1, 0, 2])
    return outputs, outputs_t[-1]

class minGRU(tf.keras.layers.Layer):
    """ Minimal Gated Recurrent Unit"""
    def __init__(
            self,
            output_size=32,
            batch_szie=1, **kwargs):
        super().__init__(**kwargs)
        self.units = output_size
        self.batch_size = batch_szie

    def build(self, input_shape):
        """ Initialize weights and biases"""
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units * 2),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel")
        self.bias = self.add_weight(
            shape=(self.units * 2,),
            initializer='zeros',
            trainable=True,
            name="bias")
        self.states = tf.zeros((input_shape[0], self.units))

    def call(
            self, inputs,
            states=None,
            return_states=False,
            training=False):
        """ Forward pass"""

        if states is None:
            states = self.states
        tmp = tf.matmul(inputs, self.kernel) + self.bias
        z, h_bar = tf.split(tmp, 2, axis=-1)
        z = tf.sigmoid(z)

        # h(t) = (1 − z(t)) ⊙ h(t−1) + z(t) ⊙ h_bar(t)
        outputs, states = scan(1.0 - z, z * h_bar, states)
        
        if return_states:
            return outputs, states
        else:
            return outputs

class minLSTM(tf.keras.layers.Layer):
    """ Minimal Long Short-Term Memory"""
    def __init__(
            self,
            output_size=32,
            batch_size=1,
            **kwargs):
        super().__init__(**kwargs)
        self.units = output_size
        self.batch_size = batch_size
        
    def build(self, input_shape):
        """ Initialize weights and biases"""
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units * 3),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel")
        self.bias = self.add_weight(
            shape=(self.units * 3,),
            initializer='zeros',
            trainable=True,
            name="bias")
        self.states = tf.zeros((input_shape[0], self.units))

    def call(self, inputs, states=None, return_state=False):
        """ Forward pass"""
        if states is None:
            states = self.states
        tmp = tf.matmul(inputs, self.kernel) + self.bias
        f, i, h_bar = tf.split(tmp, 3, axis=-1)
        f = tf.sigmoid(f)
        i = tf.sigmoid(i)

        tmp = f+i
        fp = f / tmp
        ip = 1 - fp

        outputs, states = scan(fp, ip * h_bar, states)
        if return_state:
            return outputs, states
        else:
            return outputs

if __name__ == "__main__":
    batch_size = 32
    time_step = 512
    dim_feat = 80
    width = 64
    inputs = tf.random.normal((batch_size, time_step, dim_feat))

    # gru = minGRU(output_size=width)
    # states = tf.random.normal((batch_size, width))
    # outputs, states_udpate = gru(inputs, states)
    
    
    class MyModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.gru1 = minGRU(output_size=10)
            self.gru2 = minGRU(output_size=8)

        def call(self, inputs):
            x = self.gru1(inputs)
            x = self.gru2(x)
            return x
    model = MyModel()
    outputs = model(inputs)

    print("output shape = ")
    print(outputs.shape)
    
    
    
    elems = tf.constant([1, 2, 3, 4, 5, 6])
    sum = tf.scan(lambda a, x: a + x, elems, 0)
    print(sum)
    # sum == [1, 3, 6, 10, 15, 21]
    sum = tf.scan(lambda a, x: a + x, elems, reverse=True)
    # sum == [21, 20, 18, 15, 11, 6]
    # print("states shape = ")
    # print(states_udpate.shape)
    