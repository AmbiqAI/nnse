"""
Convert numpy array to tfrecords
"""
import os
import numpy as np
import tensorflow as tf

def make_tfrecord(  fname,
                    feat,
                    target,
                    timesteps,
                    idx_starts,
                    widths):
    """
    Make tfrecord
    """
    # timesteps, dim_feat = feat.shape
    feat = feat.reshape([-1])
    writer = tf.io.TFRecordWriter(fname)
    step_feature        = tf.train.Feature(int64_list = tf.train.Int64List(value = [timesteps]))
    target_feature      = tf.train.Feature(int64_list = tf.train.Int64List(value = target))
    feat_feature        = tf.train.Feature(float_list = tf.train.FloatList(value = feat))
    idx_start_feature   = tf.train.Feature(int64_list = tf.train.Int64List(value = idx_starts))
    width_feature       = tf.train.Feature(int64_list = tf.train.Int64List(value = widths))
    context = tf.train.Features(feature = {
            "length"    : step_feature,
        })

    feature_lists = tf.train.FeatureLists(feature_list={
            "feat"          : tf.train.FeatureList(feature = [feat_feature]),
            "target"        : tf.train.FeatureList(feature = [target_feature]),
            "idx_start"     : tf.train.FeatureList(feature = [idx_start_feature]),
            "width"         : tf.train.FeatureList(feature = [width_feature]),
        })

    seq_example = tf.train.SequenceExample( # context and feature_lists
        context = context,
        feature_lists = feature_lists,
    )

    serialized = seq_example.SerializeToString()
    writer.write(serialized)
    writer.close()

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

def parser( example_proto, dim_feat):
    """
    Create a description of the features.
    """
    context_features = {
        'length'    : tf.io.FixedLenFeature([], tf.int64),
    }

    sequence_features = {
        'feat'      : tf.io.VarLenFeature(tf.float32),
        'target'    : tf.io.VarLenFeature(tf.int64),
        'idx_start' : tf.io.VarLenFeature(tf.int64),
        'width'     : tf.io.VarLenFeature(tf.int64),
    }
    context_parsed, seq_parsed = tf.io.parse_single_sequence_example(
            example_proto,
            context_features  = context_features,
            sequence_features = sequence_features,
                                        )

    # with tf.device('/device:GPU:0'):
    length = tf.cast(context_parsed['length'], tf.int32)

    target = tf.cast(tf.sparse.to_dense(seq_parsed['target']), tf.int32)
    target = tf.reshape(target, (-1,))

    idx_start = tf.cast(tf.sparse.to_dense(seq_parsed['idx_start']), tf.int32)
    idx_start = tf.reshape(idx_start, (-1,))

    width = tf.cast(tf.sparse.to_dense(seq_parsed['width']), tf.int32)
    width = tf.reshape(width, (-1,))

    label = make_target_template(target, idx_start, width, length)

    feat = tf.sparse.to_dense(seq_parsed['feat'])

    feat = tf.reshape(feat, [-1, dim_feat])


    mask = label * 0 + 1
    mask = mask[..., tf.newaxis]
    mask = tf.cast(mask, tf.float32)
    return feat, mask, label, length

def tfrecords_pipeline(
            filenames,
            dim_feat,
            batchsize = 2,
            is_shuffle = False):
    """
    Tfrecord generator
    """
    def mapping(record):
        parser_out = parser(
                    record,
                    dim_feat)
        return parser_out

    def tfrecord_convert(val):
        return tf.data.TFRecordDataset(val)

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if is_shuffle:
        dataset = dataset.shuffle(len(filenames), reshuffle_each_iteration=True)
    dataset = dataset.interleave(
                map_func           = tfrecord_convert,
                cycle_length       = tf.data.experimental.AUTOTUNE,
                block_length       = 1,
                deterministic      = False,
                num_parallel_calls = tf.data.experimental.AUTOTUNE
                                 )
    dataset = dataset.map(mapping,
                          num_parallel_calls = tf.data.experimental.AUTOTUNE
                          )
    dataset = dataset.padded_batch(batchsize)
    dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    iterator = iter(dataset)
    return iterator, dataset

def main():
    """
    Nodule testing
    """
    folder = 'tfrecord'
    os.makedirs(folder, exist_ok=True)
    dim_inputs = 100
    num_data = 53

    # generate tfrecords
    fnames = []
    for i in range(num_data):
        fname = f'{folder}/seq_test{i}.tfrecord'
        fnames += [fname]
        timesteps = np.random.randint(90, 100, (1,))[0]
        feat  = np.random.randn(timesteps, dim_inputs).astype(np.float32)
        target  = [1,0]

        starts = np.array([2, 15])
        widths = np.array([10, 3])
        target = np.array(target)

        make_tfrecord(
            fname,
            feat,
            target,
            timesteps,
            starts,
            widths)
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
                    dim_inputs,
                    batchsize = 5)

    for epoch in range(3):
        for batch, data  in enumerate(dataset):
            target = data
            print(f"Epoch {epoch}, Batch {batch}")
            print(length)
            print('feat shape:', feat.shape)
            print('mask shape:', mask.shape)
            print('target shape:', target.shape)

            print("")

if __name__ == "__main__":
    main()
