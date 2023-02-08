"""
Fake quantization module for tensorflow
"""
import tensorflow as tf
def fake_quantization(data,bitwidth,nfrac):
    """
    Fake quantization
    """
    max_val = tf.pow(2.0, bitwidth - nfrac - 1) - tf.pow(2.0, -nfrac)
    min_val = -tf.pow(2.0, bitwidth - nfrac - 1)
    out = tf.quantization.fake_quant_with_min_max_vars(
                    data,
                    min = min_val,
                    max = max_val,
                    num_bits = bitwidth)

    data.assign(out)

def get_frac_bit(data, bitwith, nfrac):
    """
    get the fractional bit
    """
    abs_data = tf.math.reduce_max(tf.abs(data))
    qbits_val = tf.math.ceil(tf.maximum(tf.math.log(abs_data)/ tf.math.log(2.0) , 0.0))
    nfrac.assign(tf.math.round(bitwith - qbits_val -1))

    return nfrac

def limiter(data, min_max):
    """
    limit the weight value
    """
    min_val, max_val = min_max
    data.assign(tf.minimum(tf.maximum(data, min_val), max_val))
