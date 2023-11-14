"""
Convert the tensorflow NN weight table to C code
"""
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Callable
from nnsp_pack.nn_module import NeuralNetClass
from nnsp_pack.load_nn_arch import load_nn_arch, setup_nn_folder

DIM_INPUT = 72  # size of a single sample, e.g. an embedding of size 64
def model() -> tf.keras.Model:
    """Example of a simple single-directional LSTM. Embedding layers are not
    supported and will have to be added as pre-processing steps manually. See the
    official TFLite documentation for more information and constraints:
    https://www.tensorflow.org/lite/models/convert/rnn. See in-line comments below
    for recommendations about tf.keras.layers.LSTM()."""

    # Define the inputs of net
    in_layer = tf.keras.layers.Input((6, DIM_INPUT, 1), batch_size=1)
    h_state = tf.keras.layers.Input(72, batch_size=1)
    c_state = tf.keras.layers.Input(72, batch_size=1)

    # Define the layers of net
    cnn_layer = tf.keras.layers.Conv2D(
                72,
                (6, 72),
                padding     = 'valid',
                strides     = (1, 1), # downsampling 2 in timesteps dim
                activation  = "tanh") # (time, dim_feat, ch)

    lstm_layer = tf.keras.layers.LSTM(
            units=72,
            return_sequences=False,
            return_state=True,
            stateful=False,
            unroll=True)
    dense_layer0 = tf.keras.layers.Dense(72, activation='tanh')
    dense_layer1 = tf.keras.layers.Dense(72, activation='tanh')
    dense_layer2 = tf.keras.layers.Dense(257, activation='sigmoid')

    # NN inference
    x = in_layer
    x = cnn_layer(x)
    x, h_state1, c_state1 = lstm_layer(x[:,:,0,:], initial_state=(h_state, c_state))
    x = dense_layer0(x)
    x = dense_layer1(x)
    out = dense_layer2(x)

    net =  tf.keras.Model(
            [in_layer, h_state, c_state],
            [out, h_state1, c_state1],
            name="LSTM")
    return net

def convert_model(model: tf.keras.Model, dataset_gen: Callable, nbit=8) -> bytes:
    """ tflite converter

    Args:
        model (tf.keras.Model): _description_
        dataset_gen (Callable): _description_
        nbit (int, optional): _description_. Defaults to 8.

    Returns:
        bytes: _description_
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = dataset_gen

    if nbit==8:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else: # nbit==16:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        converter.inference_input_type = tf.int16
        converter.inference_output_type = tf.int16
    return converter.convert()

def get_trained_model(args):
    """ get the already trained model

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    epoch_loaded    = int(args.epoch_loaded)
    nn_arch         = args.nn_arch

    out = load_nn_arch(f"{nn_arch}.txt")
    neurons, _, layer_types, activations, num_context, num_dnsampl, scalar_output, len_filter, len_lookahead = out # pylint: disable=line-too-long
    folder_nn = setup_nn_folder(nn_arch)
    nn_infer = NeuralNetClass(
        neurons     = neurons,
        layer_types = layer_types,
        activations = activations,
        batchsize   = 1,
        nDownSample = num_dnsampl,
        kernel_size = num_context)

    nn_infer.load_weights(
            f'{folder_nn}/checkpoints/model_checkpoint_ep{epoch_loaded}' )

    # nn_infer.quantized_weight()

    # with open(os.path.join(f"{folder_nn}",'stats.pkl'), "rb") as file:
    #     stats = pickle.load(file)
    return nn_infer

def main(args):
    """
    main function to convert tensorflow Neural net model to c table
    """

    net = model() # generate the model for tflite conversion
    in_data = np.random.randn(1 , 6, DIM_INPUT, 1) # (batch, timesteps, dim, ch)
    h_state = np.random.randn(1,72)
    c_state = np.random.randn(1,72)
    out_data, h_state_new, c_state_new = net([in_data,h_state, c_state]) # run the test once
    print(out_data.shape)
    print(h_state_new.shape)
    print(c_state_new.shape)
    def dataset_example(num_samples: int = 100):
        """Placeholder for a representative data-set. For best quantization
        performance, replace this with a few examples from your own data-set, the
        more the better. This should include any pre-processing needed."""
        shape_data = (1, 6, 72,1)
        shape_state = (1, 72) # lstm has 16 neurons
        for _ in range(num_samples):
            yield [tf.random.uniform(
                        shape_data,
                        minval= -2.0**15 / 2**8,
                        maxval=(2.0**15-1) / 2**8),
                    tf.random.uniform(
                        shape_state,
                        minval=-1.0,
                        maxval=1.0-2**-15),
                    tf.random.uniform(
                        shape_state,minval=-2**16,
                        maxval=(2**31-1)/2**15)]

    net_trained = get_trained_model(args) # load already trained model

    # put weight table from already trained model to new defined model
    for v in zip(net_trained.trainable_variables, net.trainable_variables):
        v1, v2 = v
        v2.assign(v1)
        print(v1.shape)
        print(v2.shape)
        print("-----------------")
    net_tflite = convert_model(net, dataset_example, nbit=args.nbit)
    Path(f"LSTM_{args.nbit}x8bit.tflite").write_bytes(net_tflite)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='Convert trained Tensorflow model to C table')

    argparser.add_argument(
        '-a',
        '--nn_arch',
        default='nn_arch/def_se_nn_arch72_mel',
        help='nn architecture')

    argparser.add_argument(
        '--epoch_loaded',
        default= 50,
        help='starting epoch')

    argparser.add_argument(
        '--nbit',
        default= 8,
        help='nbit for weight')

    argparser.add_argument(
        '--net_id',
        default= 3,
        help='starting epoch')

    argparser.add_argument(
        '--folder_c',
        default= "../evb/src",
        type=str,
        help='C folder')

    argparser.add_argument(
        '--net_name',
        default= 'se1',
        help='starting epoch')

    main(argparser.parse_args())
