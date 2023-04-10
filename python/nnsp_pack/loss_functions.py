"""
Module to calculate different loss functions
"""
import tensorflow as tf
def cross_entropy(target,       # trget
                  estimation,   # estimator
                  masking,      # mask
                  eps = tf.constant(2**-15, dtype = tf.float32)): # epsilon
    """
    Cross entropy loss per step
    """
    loss = -tf.reduce_sum(
        masking * target * tf.math.log(tf.math.maximum(estimation, eps)) )
    steps = tf.reduce_sum(masking)
    ave_loss = loss / steps
    return ave_loss, steps

def loss_mse(
        target,       # trget
        estimation,   # estimator
        masking,      # mask
        reg = 0,       # regularization,
        lam = 1.0  # coefficient of reg
        ):
    """
    MSE loss per step
    """
    mse = tf.reduce_sum(
        masking * tf.math.square(target - estimation) )
    steps = tf.reduce_sum(masking)
    ave_loss = mse / steps + lam * reg
    return ave_loss, steps

def loss_sdr(clean,
             est,
             masking,
             eps=2.0**-15):
    """
    distortion / signal
    """
    err = (clean - est)**2
    power_s = (clean) **2
    loss = tf.reduce_sum(masking * (tf.math.log(err+eps) - tf.math.log(power_s+eps)))
    steps = tf.reduce_sum(masking)
    ave_loss = loss / steps
    return ave_loss, steps
    