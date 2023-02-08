"""
Calculating statistic mean and standard deviation
"""
import os
import pickle
import tensorflow as tf
import numpy as np
from .converter_fix_point import fakefix_tf
from .tf_basic_math import tf_log10_eps

def feat_stats_estimator(
        dataset,
        fnames,
        batchsize,
        dim_feat,
        folder_nn,
        feat_type='mel'):
    """
    Estimate statistics of training data
    """
    fbanks = np.load('fbank_mel.npy')
    fbanks = tf.Variable(fbanks.T, dtype=tf.float32, trainable = False)
    mean_stats = tf.Variable( tf.zeros((dim_feat,), dtype = tf.float64),
                    dtype = tf.float64, trainable = False)
    inv_std_stats = tf.Variable(tf.zeros((dim_feat,), dtype = tf.float64),
                        dtype = tf.float64, trainable = False)
    tot = tf.Variable(0, dtype = tf.float64)

    num_batches = int(len(fnames) / batchsize)

    def convert_pspec2melspec(data, feat_type='mel'):
        """
        more feat extraction
        """
        if feat_type=='mel':
            feats = tf.matmul(data, fbanks)
        elif feat_type=='pspec':
            feats = tf_log10_eps(data)
        return fakefix_tf(feats, 32, 15)

    # mean calculation
    for batch, data in enumerate(dataset):
        if batch % 5 == 0:
            tf.print(f"\rMean estimating (batch) {int(batch / 5)}/{num_batches}, ",
                        end = '')
        pspec_sn, masks, _, _ = data
        feats = convert_pspec2melspec(pspec_sn, feat_type)
        _, _, dim_feat = feats.shape
        tmp = tf.math.reduce_sum(feats * masks, axis = (0,1))
        mean_stats = mean_stats + tf.cast(tmp, tf.float64)
        tmp = tf.math.reduce_sum(masks)
        tot = tot + tf.cast(tmp, tf.float64)

    mean_stats = mean_stats / tot
    mean_stats = tf.cast(mean_stats, tf.float32)
    mean_stats = fakefix_tf(mean_stats, 32, 15)

    # std calculation
    for batch, data in enumerate(dataset):
        if batch % 5 == 0:
            tf.print(f"\rSTD estimating (batch) {int(batch / 5)}/{num_batches}, ",
                        end = '')
        pspec_sn, masks, _, _ = data
        feats = convert_pspec2melspec(pspec_sn, feat_type)
        _, _, dim_feat = feats.shape
        tmp = tf.math.reduce_sum( masks * (feats - mean_stats)**2, axis = (0,1))
        inv_std_stats = inv_std_stats + tf.cast(tmp, tf.float64)

    inv_std_stats = 1.0 / tf.math.sqrt(inv_std_stats / tot)
    inv_std_stats = tf.cast(inv_std_stats, tf.float32)
    inv_std_stats = fakefix_tf(inv_std_stats, 32, 15)

    # save mean and std
    stats = {'nMean_feat': mean_stats.numpy(), 'nInvStd': inv_std_stats.numpy()}

    stats_name = 'stats.pkl'
    with open(os.path.join(folder_nn, stats_name), "wb") as file:
        pickle.dump(stats, file)

    return stats
