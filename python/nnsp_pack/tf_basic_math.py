"""
Basic tensorflow math functions
"""
import tensorflow as tf
def tf_log10_eps(val, eps = 2.0**-15):
    """
    log10 with minimum eps
    """
    return  tf.math.log(val+eps) / tf.math.log(10.0)

def tf_power_eps(val, pwr = 0.3, eps = 2.0**-15):
    """
    val**pwr with minimum eps
    """
    return (eps+val)**pwr
