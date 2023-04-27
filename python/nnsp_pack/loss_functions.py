"""
Module to calculate different loss functions
"""
import tensorflow as tf
import numpy as np

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

def deepfiltering(
        amp_sn,
        tfmask):
    """Deep filtering
    shape = (batchsize, timesteps, dim_feat)

    Args:
        amp_sn (tf.float32): amplitude of noisy signal
        tfmask (tf.float32): estimated tfmask
    """
    eps = 2**-15
    batchsize, steps, dim_feat = tfmask.shape
    shape = (batchsize,steps, 4, dim_feat >> 2)
    tfmask = tf.reshape(tfmask, shape)
    amp_extend=[]
    pad = tf.ones((batchsize, 3, dim_feat >> 2), dtype = np.float32) * eps
    pad_amp = tf.concat([pad, tf.identity(amp_sn)], axis=-2)
    for i in range(4):
        amp_extend +=[tf.identity(pad_amp[:,i:i+steps,:])]
    amp_extend = tf.stack(amp_extend,-2)
    estimation = tf.reduce_mean(amp_extend * tfmask, axis=-2)
    ave_mask = tf.reduce_mean(tfmask, axis=-2)

    return estimation, ave_mask

def deepfiltering_np(
        amp_sn,
        tfmask):
    """Deep filtering for numpy
    shape = (timesteps, dim_feat), No batchsize

    Args:
        amp_sn (np.float32): amplitude of noisy signal
        tfmask (np.float32): estimated tfmask
    """
    eps=2**-15
    steps, dim_feat = tfmask.shape
    shape = (steps, 4, dim_feat >> 2)
    tfmask = np.reshape(tfmask, shape)
    amp_extend=[]
    pad = np.ones((3, dim_feat >> 2), dtype = np.float32) * eps
    pad_amp = np.concatenate([pad, amp_sn.copy()], axis=-2)
    for i in range(4):
        amp_extend +=[pad_amp[i:i+steps,:].copy()]
    amp_extend = np.stack(amp_extend,-2)
    estimation = np.mean(amp_extend * tfmask, axis=-2)
    ave_mask = np.mean(tfmask, axis=-2)

    return estimation, ave_mask

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
    