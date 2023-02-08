"""
Fixed point converter module
"""
import numpy as np
import tensorflow as tf

def fakefix(val, bit_width, bit_frac):
    """
    fake fixed point for numpy
    """
    tmp = 1 << (bit_width - 1)
    max_val = tmp - 1
    min_val = -tmp
    out = np.floor(val * 2**bit_frac)
    return np.minimum(np.maximum(out, min_val), max_val) / 2**bit_frac

def fakefix_tf(val_in, nbit, qbit):
    """
    fake fixed point for tensor
    """
    min_val = -2**(nbit-1)
    max_val = 2**(nbit-1)-1

    out = tf.math.floor(val_in * 2**qbit)
    return tf.math.minimum(tf.math.maximum(out, min_val), max_val) / 2**qbit

def cplx_fakefix(val, bit_width, bit_frac):
    """
    complex fake fixed point for numpy
    """
    real = fakefix(np.real(val),bit_width, bit_frac)
    imag = fakefix(np.imag(val),bit_width, bit_frac)
    return real + 1j * imag

def float2hex(val_in):
    """
    float point to hex
    """
    if val_in < 0:
        val_in = -val_in
        sign = -1
    else:
        sign = 1

    if sign == -1:
        out = 2**16 - int(val_in*2**15)
    else:
        out = int(val_in*2**15)
    return out
