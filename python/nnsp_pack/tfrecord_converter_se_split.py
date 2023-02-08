"""
Convert numpy array to tfrecords
"""
import os
import numpy as np
import tensorflow as tf

def make_tfrecord(
        fname,
        pspec_sn_raw,
        pspec_s_raw):
    """
    Make tfrecord
    """
    with tf.io.TFRecordWriter(fname) as writer:

        pspec_sn_list = np.split(pspec_sn_raw, 5, axis = 0)
        pspec_s_list  = np.split(pspec_s_raw, 5, axis = 0)

        for pspec_sn, pspec_s in zip(pspec_sn_list, pspec_s_list):
            timesteps, _ = pspec_s.shape
            pspec_sn = pspec_sn.reshape([-1])
            pspec_s  = pspec_s.reshape([-1])

            step_feature = tf.train.Feature(
                int64_list = tf.train.Int64List(value = [timesteps]))

            pspec_sn_feature = tf.train.Feature(
                float_list = tf.train.FloatList(value = pspec_sn))

            pspec_s_feature = tf.train.Feature(
                float_list = tf.train.FloatList(value = pspec_s))

            context = tf.train.Features(feature = {
                    "length"    : step_feature,
                })

            feature_lists = tf.train.FeatureLists(feature_list={
                    "pspec_sn" : tf.train.FeatureList(feature = [pspec_sn_feature]),
                    "pspec_s"  : tf.train.FeatureList(feature = [pspec_s_feature]),
                })

            seq_example = tf.train.SequenceExample( # context and feature_lists
                context = context,
                feature_lists = feature_lists,
            )

            serialized = seq_example.SerializeToString()
            writer.write(serialized)

def make_target_template(targets, idx_starts, widths, steps):
    """
    Make a rectangular function for a template
    """
    out = tf.zeros((steps,), dtype=tf.int32)
    for i in range(2):
        width = widths[i]
        idx_start = idx_starts[i]
        target  = targets[i]
        start   = tf.zeros((idx_start, ), dtype=tf.int32)
        middle  = tf.ones((width,), dtype=tf.int32) * target
        ending  = tf.zeros((steps - idx_start - width,), dtype=tf.int32)
        concat  = tf.concat( (start, middle, ending), 0 )
        out = out + tf.cast(concat, tf.int32)
    return out

def parser( example_proto):
    """
    Create a description of the features.
    """
    context_features = {
        'length'    : tf.io.FixedLenFeature([], tf.int64),
    }

    sequence_features = {
        'pspec_sn'      : tf.io.VarLenFeature(tf.float32),
        'pspec_s'       : tf.io.VarLenFeature(tf.float32),
    }
    context_parsed, seq_parsed = tf.io.parse_single_sequence_example(
            example_proto,
            context_features  = context_features,
            sequence_features = sequence_features,
                                        )

    # with tf.device('/device:GPU:0'):
    length = tf.cast(context_parsed['length'], tf.int32)

    pspec_sn = tf.sparse.to_dense(seq_parsed['pspec_sn'])
    pspec_sn = tf.reshape(pspec_sn, [-1, 257])

    pspec_s = tf.sparse.to_dense(seq_parsed['pspec_s'])
    pspec_s = tf.reshape(pspec_s, [-1, 257])

    mask = pspec_s[:,0] * 0 + 1
    mask = mask[..., tf.newaxis]
    mask = tf.cast(mask, tf.float32)

    return pspec_sn, mask, pspec_s, length

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
    dataset = dataset.batch(
                    batchsize,
                    drop_remainder=True)
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
                    num_parallel_calls = tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.threading.max_intra_op_parallelism = 1
    options.threading.private_threadpool_size = 6
    dataset = dataset.with_options(options)
    dataset = dataset.prefetch(buffer_size = tf.data.AUTOTUNE)
    iterator = iter(dataset)
    return iterator, dataset

def main():
    """
    Nodule testing
    """
    folder = 'tfrecord'
    os.makedirs(folder, exist_ok=True)
    num_data = 53

    # generate tfrecords
    fnames = []
    for i in range(num_data):
        fname = f'{folder}/seq_test{i}.tfrecord'
        fnames += [fname]

        lst = []
        for k in range(5):
            lst += [np.ones((500, 257), dtype = np.float32) * i + k * 0.1]
        pspec_sn = np.concatenate(lst, axis=0)
        pspec_s = pspec_sn.copy()
        make_tfrecord(
            fname,
            pspec_sn,
            pspec_s)
        print(f"\r{i}", end = '')

    with open('data_list.txt', 'w') as file: # pylint: disable=unspecified-encoding
        for fname in fnames:
            file.write(f'{fname}\n')

    # scan batches of tfrecord data
    fnames = []
    for i in range(num_data):
        fname = f'{folder}/seq_test{i}.tfrecord'
        fnames += [fname]
    _, dataset = tfrecords_pipeline(
                    fnames,
                    batchsize = 5,
                    is_shuffle=True)

    for epoch in range(3):
        for batch, data  in enumerate(dataset):
            pspec_sn, _, pspec_s, _ = data
            if batch in {0,1,2,3,4}:
                print(f"Epoch {epoch}, Batch {batch}")
                print('pspec_sn shape:', pspec_sn.shape)
                print('pspec_s shape:', pspec_s.shape)
                for i in range(5):
                    print(pspec_sn[i][0,0:5])
                print("")

if __name__ == "__main__":
    main()
