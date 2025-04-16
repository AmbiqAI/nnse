"""
Convert numpy array to tfrecords
"""
import os
import numpy as np
import tensorflow as tf

def make_raw_tfrecord(
        fname,
        audio_sn,
        audio_s):
    """
    Make tfrecord
    """
    with tf.io.TFRecordWriter(fname) as writer:

        timesteps = audio_s.shape[0]
        # pspec_sn = pspec_sn.reshape([-1])
        # pspec_s  = pspec_s.reshape([-1])

        step_feature = tf.train.Feature(
            int64_list = tf.train.Int64List(value = [timesteps]))

        audio_sn_feature = tf.train.Feature(
            float_list = tf.train.FloatList(value = audio_sn))

        audio_s_feature = tf.train.Feature(
            float_list = tf.train.FloatList(value = audio_s))

        context = tf.train.Features(feature = {
                "length"    : step_feature,
            })

        feature_lists = tf.train.FeatureLists(feature_list={
                "audio_sn" : tf.train.FeatureList(feature = [audio_sn_feature]),
                "audio_s"  : tf.train.FeatureList(feature = [audio_s_feature]),
            })

        seq_example = tf.train.SequenceExample( # context and feature_lists
            context = context,
            feature_lists = feature_lists,
        )

        serialized = seq_example.SerializeToString()
        writer.write(serialized)

def parser( example_proto):
    """
    Create a description of the features.
    """
    context_features = {
        'length'    : tf.io.FixedLenFeature([], tf.int64),
    }

    sequence_features = {
        'audio_sn'      : tf.io.VarLenFeature(tf.float32),
        'audio_s'       : tf.io.VarLenFeature(tf.float32),
    }
    context_parsed, seq_parsed = tf.io.parse_single_sequence_example(
            example_proto,
            context_features  = context_features,
            sequence_features = sequence_features,
                                        )

    # with tf.device('/device:GPU:0'):
    length = tf.cast(context_parsed['length'], tf.int32)

    audio_sn = tf.sparse.to_dense(seq_parsed['audio_sn'])
    # pspec_sn = tf.reshape(pspec_sn, [-1, 257])

    audio_s = tf.sparse.to_dense(seq_parsed['audio_s'])
    # pspec_s = tf.reshape(pspec_s, [-1, 257])

    # mask = pspec_s[:,0] * 0 + 1
    # mask = mask[..., tf.newaxis]
    # mask = tf.cast(mask, tf.float32)

    return audio_sn, audio_s, length

def tfrecords_pipeline(
            filenames,
            batchsize = 2,
            is_shuffle = False):
    """
    Tfrecord generator
    """
    def mapping(record):
        return parser(record)

    def tfrecord_convert(val):
        return tf.data.TFRecordDataset(val)

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if is_shuffle:
        dataset = dataset.shuffle(len(filenames), reshuffle_each_iteration=True)
    dataset = dataset.interleave(
                map_func           = tfrecord_convert,
                cycle_length       = batchsize,
                block_length       = 1,
                deterministic      = True,
                num_parallel_calls = tf.data.AUTOTUNE)
    dataset = dataset.map(
                mapping,
                num_parallel_calls = tf.data.AUTOTUNE,
                deterministic = True)
    dataset = dataset.batch(
                    batchsize,
                    drop_remainder=True,
                    num_parallel_calls = 3)
    dataset = dataset.prefetch(buffer_size = 1)
    iterator = iter(dataset)
    return iterator, dataset

def main():
    """
    Nodule testing
    """
    batchsize=2
    with open('data/test_tfrecords_se_raw.csv') as file:
        lines = file.readlines()
        fnames = [line.strip() for line in lines]
    _, dataset = tfrecords_pipeline(
                    fnames,
                    batchsize = batchsize,
                    is_shuffle=False)

    for epoch in range(2):
        batch_id = 0
        for batch, data  in enumerate(dataset):
            audio_sn, audio_s, length = data
            import pdb; pdb.set_trace()
            print(f"Epoch {epoch}, Batch {batch_id}")
            print('pspec_sn shape:', audio_sn.shape)
            print('pspec_s shape:', audio_s.shape)
            
            print("")
            
if __name__ == "__main__":
    main()
