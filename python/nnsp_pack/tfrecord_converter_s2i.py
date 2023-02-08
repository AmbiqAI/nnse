"""
Convert numpy array to tfrecords
"""
import os
import numpy as np
import tensorflow as tf

def make_tfrecord(  fname,
                    feat,
                    intent,
                    slot0,
                    slot1,
                    timesteps,
                    idx_start,
                    width):
    """
    Make tfrecord
    """
    # timesteps, dim_feat = feat.shape
    feat = feat.reshape([-1])
    writer = tf.io.TFRecordWriter(fname)
    step_feature        = tf.train.Feature(int64_list = tf.train.Int64List(value = [timesteps]))
    idx_start_feature   = tf.train.Feature(int64_list = tf.train.Int64List(value = [idx_start]))
    width_feature       = tf.train.Feature(int64_list = tf.train.Int64List(value = [width]))
    intent_feature      = tf.train.Feature(int64_list = tf.train.Int64List(value = [intent]))
    slot0_feature       = tf.train.Feature(int64_list = tf.train.Int64List(value = [slot0]))
    slot1_feature       = tf.train.Feature(int64_list = tf.train.Int64List(value = [slot1]))
    feat_feature        = tf.train.Feature(float_list = tf.train.FloatList(value = feat))

    context = tf.train.Features(feature = {
            "length":       step_feature,
            "idx_start":    idx_start_feature,
            "width":        width_feature,
            "intent":       intent_feature,
            "slot0":        slot0_feature,
            "slot1":        slot1_feature,
        })

    feature_lists = tf.train.FeatureLists(feature_list={
            "feat": tf.train.FeatureList(feature = [feat_feature]),
        })

    seq_example = tf.train.SequenceExample( # context and feature_lists
        context = context,
        feature_lists = feature_lists,
    )

    serialized = seq_example.SerializeToString()
    writer.write(serialized)
    writer.close()

def make_target_template(idx_start, width, steps):
    """
    Make a rectangular function for a template
    """
    start  = tf.zeros((idx_start, ))
    middle  = tf.ones((width,))
    ending  = tf.zeros((steps - idx_start - width,))
    concat = tf.concat( (start, middle, ending), 0 )
    out = tf.cast(concat, tf.int32)
    return out

def parser( example_proto, dim_feat):
    """
    Create a description of the features.
    """
    context_features = {
        'length':       tf.io.FixedLenFeature([], tf.int64),
        'idx_start':    tf.io.FixedLenFeature([], tf.int64),
        'width':        tf.io.FixedLenFeature([], tf.int64),
        'intent':       tf.io.FixedLenFeature([], tf.int64),
        'slot0':        tf.io.FixedLenFeature([], tf.int64),
        'slot1':        tf.io.FixedLenFeature([], tf.int64),
    }

    sequence_features = {
        'feat': tf.io.VarLenFeature(tf.float32),
    }
    context_parsed, seq_parsed = tf.io.parse_single_sequence_example(example_proto,
                                        context_features  = context_features,
                                        sequence_features = sequence_features,
                                        )

    # with tf.device('/device:GPU:0'):
    length = tf.cast(context_parsed['length'], tf.int32)

    idx_start = tf.cast(context_parsed['idx_start'], tf.int32)

    width = tf.cast(context_parsed['width'], tf.int32)

    trigger_template = make_target_template(idx_start, width, length)

    intent = tf.cast(context_parsed['intent'], tf.int32)
    slot0  = tf.cast(context_parsed['slot0'],  tf.int32)
    slot1  = tf.cast(context_parsed['slot1'],  tf.int32)
    feat = tf.sparse.to_dense(seq_parsed['feat'])
    feat = tf.reshape(feat, [-1, dim_feat])

    intent = trigger_template * intent
    # intent = tf.one_hot(intent, dim_intent)
    # intent = tf.cast(intent, tf.float32)

    slot0 = trigger_template * slot0
    # slot0 = tf.one_hot(slot0, dim_slot)
    # slot0 = tf.cast(slot0, tf.float32)

    slot1 = trigger_template * slot1
    # slot1 = tf.one_hot(slot1, dim_slot)
    # slot1 = tf.cast(slot1, tf.float32)

    mask = intent * 0 + 1
    mask = mask[..., tf.newaxis]
    mask = tf.cast(mask, tf.float32)

    return feat, mask, intent, slot0, slot1, length

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
    dim_intent = 7
    dim_slots = 17
    num_data = 53

    # generate tfrecords
    width=4
    fnames = []
    for i in range(num_data):
        fname = f'{folder}/seq_test{i}.tfrecord'
        fnames += [fname]
        timesteps = np.random.randint(90, 100, (1,))[0]
        start_idx=3

        feat  = np.random.randn(timesteps, dim_inputs).astype(np.float32)

        intent  = np.random.randint(0,dim_intent, (1,))[0]

        slot0   = np.random.randint(0,dim_slots,  (1,))[0]
        slot1   = np.random.randint(0,dim_slots,  (1,))[0]

        make_tfrecord(fname, feat, intent,
                      slot0, slot1,
                      timesteps, start_idx, width)
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
                    dim_slots,
                    batchsize = 5)

    for epoch in range(3):
        for batch, data  in enumerate(dataset):
            feat, mask, intent, slot0, slot1, length = data
            print(f"Epoch {epoch}, Batch {batch}")
            print(length)
            print('feat shape:', feat.shape)
            print('mask shape:', mask.shape)
            print('intent shape:', intent.shape)
            print('slot0 shape:', slot0.shape)
            print('slot1 shape:', slot1.shape)
            print("")

if __name__ == "__main__":
    main()
