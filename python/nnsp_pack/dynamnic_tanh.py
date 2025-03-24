import tensorflow as tf

class dynamnic_tanh(tf.keras.layers.Layer):
    """ Dynamic tanh layer
        This layer applies a dynamic tanh layer 
        to replace the layer normalization layer.
        Reference: https://arxiv.org/pdf/2503.10622
    """
    def __init__(self, **kwargs):
        """ Initialize the layer """
        super(dynamnic_tanh, self).__init__(**kwargs)

    def build(self, input_shape):
        """ Build the layer """
        if len(input_shape) ==3:
            shape=(1, 1, input_shape[-1])
        elif len(input_shape) == 4:
            shape=(1, 1, input_shape[-2], 1)
        self.alpha = self.add_weight(
            shape=(1,),
            initializer="ones",
            trainable=True,
            name='alpha') * 0.5

        self.gamma=self.add_weight(
            shape=shape,
            initializer="ones",
            trainable=True,
            name='gamma')

        self.beta = self.add_weight(
            shape=shape,
            initializer="zeros",
            trainable=True,
            name='beta')

    def call(self, inputs):
        """ Forward pass """
        x = tf.tanh(self.alpha * inputs)
        return self.gamma * x + self.beta

if __name__ == "__main__":
    # Test the dynamnic_tanh layer
    inputs = tf.random.normal((4, 5, 10))
    layer = dynamnic_tanh()
    outputs = layer(inputs)
    print("Inputs:", inputs.numpy())
    print("Outputs:", outputs.numpy())
    print("Alpha:", layer.alpha.numpy())
    print("Alpha shape:", layer.alpha.shape)
    print("Beta:", layer.beta.numpy())
    print("Beta shape:", layer.beta.shape)
    print("Gamma:", layer.gamma.numpy())
    print("Gamma shape:", layer.gamma.shape)
