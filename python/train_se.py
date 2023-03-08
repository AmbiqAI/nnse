"""
Training script for s2i RNN
"""
import os
import re
import logging
import argparse
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from nnsp_pack.nn_module import NeuralNetClass, lstm_states, tf_round
from nnsp_pack.tfrecord_converter_se_split import tfrecords_pipeline
from nnsp_pack.loss_functions import loss_mse
from nnsp_pack.converter_fix_point import fakefix_tf
from nnsp_pack.calculate_feat_stats_se_split import feat_stats_estimator
from nnsp_pack.load_nn_arch import load_nn_arch, setup_nn_folder
from nnsp_pack.tf_basic_math import tf_log10_eps, tf_power_eps
import c_code_table_converter

SHOW_STEPS          = False
DISPLAY_HISTOGRAM   = True
BLOCKS_PER_AUDIO    = 5
DIM_TARGET          = 257
physical_devices    = tf.config.list_physical_devices('GPU')

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: # pylint: disable=bare-except
    pass

MEL_FBANKS = tf.Variable(
    np.load('fbank_mel.npy').T,
    dtype       = tf.float32,
    trainable   = False)

@tf.function
def train_kernel(
        nfeat, # melspec
        pspec_sn,
        pspec_s,
        mask,
        states,
        net,
        optimizer,
        training    = True,
        quantized   = False):
    """
    Training kernel
    """
    with tf.GradientTape() as tape:
        est, states = net(
                nfeat,
                mask,
                states,
                training    = training,
                quantized   = quantized)

        amp_sn = tf.math.sqrt(pspec_sn)
        amp_s  = tf.math.sqrt(pspec_s)

        ave_loss, steps = loss_mse(
                tf_power_eps(amp_s),        # clean
                tf_power_eps(amp_sn * est), # noisy * mask
                masking = mask)

    if training:
        gradients = tape.gradient(ave_loss, net.trainable_variables)

        gradients_clips = [ tf.clip_by_norm(grad, 1)
                            for grad in gradients ]

        optimizer.apply_gradients(
                    zip(gradients_clips,
                        net.trainable_variables))

    return est, states, ave_loss, steps

def epoch_proc(
        net,
        optimizer,
        dataset,
        fnames,
        batchsize,
        timesteps,
        training,
        zero_state,
        norm_mean,
        norm_inv_std,
        num_dnsampl     = 1,
        num_context     = 6,
        quantized       = False,
        feat_type       = 'mel'):
    """
    Training for one epoch
    """
    dim_feat, = norm_mean.shape

    net.reset_stats()
    total_batches = int(len(fnames) / batchsize)
    for batch, data in enumerate(dataset):
        if batch % BLOCKS_PER_AUDIO == 0:
            states = lstm_states(net, batchsize, zero_state=zero_state)

            shape = (batchsize, num_context-1, dim_feat)

            padddings_tsteps = tf.constant(
                            np.full(shape, np.log10(2**-15)),
                            dtype = tf.float32)
        else:
            padddings_tsteps = tf.identity(feats[:,-(num_context-1):,:])

        pspec_sn, masks, pspec_s, _ = data
        if feat_type == 'mel':
            feats = tf.matmul(pspec_sn, MEL_FBANKS)
        elif feat_type == 'pspec':
            feats = tf.identity(pspec_sn)
        feats = tf_log10_eps(feats)
        feats = fakefix_tf(feats, 32, 15)
        feats = tf.concat([padddings_tsteps, feats], 1)
        nfeats = (feats - norm_mean) * norm_inv_std
        nfeats = fakefix_tf(nfeats, 16, 8)
        _, steps, _ = pspec_sn.shape
        for k in range(steps // timesteps):
            start = k * timesteps
            end = (k+1) * timesteps
            tmp = train_kernel(
                    tf.identity(  nfeats[:,start:end+num_context-1,:]),
                    tf.identity(pspec_sn[:,start:end:num_dnsampl,:]),
                    tf.identity( pspec_s[:,start:end:num_dnsampl,:]),
                    tf.identity(   masks[:,start:end:num_dnsampl,:]),
                    states,
                    net,
                    optimizer,
                    training    = training,
                    quantized   = quantized)

            _, states, ave_loss, steps = tmp

        net.update_cost_steps(ave_loss, steps)

        if batch % BLOCKS_PER_AUDIO == (BLOCKS_PER_AUDIO-1):
            tf.print(f"\r {int(batch / 5)}/{total_batches}: ",
                        end = '')
            net.show_loss(
                net.stats['acc_loss'],
                net.stats['acc_matchCount'],
                net.stats['acc_steps'],
                SHOW_STEPS)

        # Debugging
        if 0: # pylint: disable=using-constant-test
            idx = 10
            if batch % BLOCKS_PER_AUDIO == 0:
                feat = feats[idx,:,:].numpy()
                logspec_s = tf_log10_eps(pspec_s[idx,:,:]).numpy()
                logspec_sn = tf_log10_eps(pspec_sn[idx,:,:]).numpy()
            else:
                feat = np.concatenate((feat, feats[idx,:,:].numpy()), axis=0)
                logspec_s = np.concatenate(
                                (logspec_s, tf_log10_eps(pspec_s[idx,:,:]).numpy()),
                                axis=0)
                logspec_sn = np.concatenate(
                                (logspec_sn, tf_log10_eps(pspec_sn[idx,:,:]).numpy()),
                                axis=0)

            if batch % BLOCKS_PER_AUDIO == (BLOCKS_PER_AUDIO-1):
                plt.figure(1)
                plt.clf()

                plt.subplot(3,1,1)
                plt.imshow(
                    feat.T,
                    origin      = 'lower',
                    cmap        = 'pink_r',
                    aspect      = 'auto')

                plt.subplot(3,1,2)
                plt.imshow(
                    logspec_sn.T,
                    origin      = 'lower',
                    cmap        = 'pink_r',
                    aspect      = 'auto')

                plt.subplot(3,1,3)
                plt.imshow(
                    logspec_s.T,
                    origin      = 'lower',
                    cmap        = 'pink_r',
                    aspect      = 'auto')

                plt.show()
    tf.print('\n', end = '')

def main(args):
    """
    main function to train neural network training
    """
    batchsize       = args.batchsize
    timesteps       = args.timesteps
    num_epoch       = args.num_epoch
    epoch_loaded    = args.epoch_loaded
    quantized       = args.quantized

    tfrecord_list = {   'train' : args.train_list,
                        'test'  : args.test_list}

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    arch = load_nn_arch(args.nn_arch)
    neurons, drop_rates, layer_types, activations, num_context, num_dnsampl = arch

    folder_nn = setup_nn_folder(args.nn_arch)

    dim_feat = neurons[0]

    nn_train = NeuralNetClass(
        neurons     = neurons,
        layer_types = layer_types,
        dropRates   = drop_rates,
        activations = activations,
        batchsize   = batchsize,
        nDownSample = num_dnsampl,
        kernel_size = num_context,
        dim_target  = DIM_TARGET)

    nn_infer = NeuralNetClass(
        neurons     = neurons,
        layer_types = layer_types,
        activations = activations,
        batchsize   = batchsize,
        nDownSample = num_dnsampl,
        kernel_size = num_context,
        dim_target  = DIM_TARGET)

    if epoch_loaded == 'random':
        epoch_loaded = -1

        loss = {'train' : np.zeros(num_epoch+1),
                'test'  : np.zeros(num_epoch+1)}

        acc  = {'train' : np.zeros(num_epoch+1),
                'test'  : np.zeros(num_epoch+1)}

        epoch1_loaded = epoch_loaded + 1
    else:
        if epoch_loaded == 'latest':
            checkpoint_dir = f'{folder_nn}/checkpoints'
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            nn_train.load_weights(latest)
            tmp = re.search(r'_ep(\d)+', latest)
            epoch_loaded = int(re.sub(r'_ep','',tmp.group(0)))
            epoch1_loaded = epoch_loaded + 1
        else:
            nn_train.load_weights(
                f'{folder_nn}/checkpoints/model_checkpoint_ep{epoch_loaded}')
            epoch1_loaded = epoch_loaded + 1

        print(f"Model at epoch {epoch1_loaded - 1} is retrieved")

        with open(os.path.join(folder_nn, 'nn_loss.pkl'), "rb") as file:
            loss = pickle.load(file)
        with open(os.path.join(folder_nn, 'nn_acc.pkl'), "rb") as file:
            acc = pickle.load(file)

        ax_handle = plt.subplot(2,1,1)
        ax_handle.plot(loss['train'][0: epoch_loaded+1])
        ax_handle.plot(loss['test'][0: epoch_loaded+1])
        ax_handle.legend(['train', 'test'])
        ax_handle.grid(True)
        ax_handle.set_title(f'Loss and accuracy upto epoch {epoch_loaded}. Close it to continue')

        print(f"(train) best epoch picked by loss = {np.argmin(loss['train'][0: epoch_loaded+1])}")
        print(f"(test)  best epoch picked by loss = {np.argmin(loss['test'][0: epoch_loaded+1])}")

        ax_handle.grid(True)

        plt.show()

    nn_train.duplicated_to(nn_infer, logger)

    fnames = {}
    for tr_set in ['train', 'test']:
        with open(tfrecord_list[tr_set], 'r') as file: # pylint: disable=unspecified-encoding
            try:
                lines = file.readlines()
            except:# pylint: disable=bare-except
                print(f'Can not find the list {tfrecord_list[tr_set]}')
            else:
                len0 = int(len(lines) / batchsize) * batchsize
                # len0 = batchsize * 10
                fnames[tr_set] = [line.strip() for line in lines[:len0]]

    _, dataset = tfrecords_pipeline(
            fnames['train'],
            batchsize = batchsize,
            is_shuffle = True)

    _, dataset_tr = tfrecords_pipeline(
            fnames['train'],
            batchsize = batchsize,
            is_shuffle = False)

    _, dataset_te = tfrecords_pipeline(
            fnames['test'],
            batchsize = batchsize,
            is_shuffle = False)


    if os.path.exists(f'{folder_nn}/stats.pkl'):
        with open(os.path.join(folder_nn, 'stats.pkl'), "rb") as file:
            stats = pickle.load(file)
    else:
        stats = feat_stats_estimator(
                dataset_tr, fnames['train'],
                batchsize, dim_feat, folder_nn,
                feat_type=args.feat_type)

    nn_np = c_code_table_converter.tf2np(nn_infer, quantized=quantized)
    if DISPLAY_HISTOGRAM:
        c_code_table_converter.draw_nn_hist(nn_np)
        c_code_table_converter.draw_nn_weight(
            nn_np,
            nn_infer,
            pruning=False)

    for epoch in range(epoch1_loaded, num_epoch):
        t_start = tf.timestamp()
        tf.print(f'\n(EP {epoch})\n', end = '')

        # Training phase
        if 1:
            epoch_proc(
                nn_train,
                optimizer,
                dataset,
                fnames['train'],
                batchsize,
                timesteps,
                training        = True,
                zero_state      = False,
                norm_mean       = stats['nMean_feat'],
                norm_inv_std    = stats['nInvStd'],
                num_dnsampl     = num_dnsampl,
                num_context     = num_context,
                quantized       = quantized,
                feat_type       = args.feat_type)

        nn_train.duplicated_to(
                nn_infer,
                logger)

        # Computing Training loss
        epoch_proc(
            nn_infer,
            optimizer,
            dataset_tr,
            fnames['train'],
            batchsize,
            timesteps,
            training        = False,
            zero_state      = True,
            norm_mean       = stats['nMean_feat'],
            norm_inv_std    = stats['nInvStd'],
            num_dnsampl     = num_dnsampl,
            num_context     = num_context,
            quantized       = quantized,
            feat_type       = args.feat_type)

        loss['train'][epoch] = nn_infer.stats['acc_loss'] / nn_infer.stats['acc_steps']
        loss['train'][epoch] /= nn_infer.neurons[-1]

        acc['train'][epoch] = nn_infer.stats['acc_matchCount'] / nn_infer.stats['acc_steps']

        # Computing Testing loss
        epoch_proc(
            nn_infer,
            optimizer,
            dataset_te,
            fnames['test'],
            batchsize,
            timesteps,
            training            = False,
            zero_state          = True,
            norm_mean           = stats['nMean_feat'],
            norm_inv_std        = stats['nInvStd'],
            num_dnsampl         = num_dnsampl,
            num_context         = num_context,
            quantized           = quantized,
            feat_type           = args.feat_type)

        loss['test'][epoch] = nn_infer.stats['acc_loss'] / nn_infer.stats['acc_steps']
        loss['test'][epoch] /= nn_infer.neurons[-1]

        acc['test'][epoch] = nn_infer.stats['acc_matchCount'] / nn_infer.stats['acc_steps']

        nn_train.save_weights(f'{folder_nn}/checkpoints/model_checkpoint_ep{epoch}')

        with open(os.path.join(folder_nn, 'nn_loss.pkl'), "wb") as file:
            pickle.dump(loss, file)
        with open(os.path.join(folder_nn, 'nn_acc.pkl'), "wb") as file:
            pickle.dump(acc, file)

        tf.print('Epoch spent ', tf_round(tf.timestamp() - t_start), ' seconds')

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(
        description='Training script for se model')

    argparser.add_argument(
        '-a',
        '--nn_arch',
        default='nn_arch/def_se_nn_arch.txt',
        help='nn architecture')

    argparser.add_argument(
        '-ft',
        '--feat_type',
        default='mel',
        help='feature type: \'mel\'or \'pspec\'')

    argparser.add_argument(
        '-tr',
        '--train_list',
        default='data/train_tfrecords_se.csv',
        help='train_list')

    argparser.add_argument(
        '-te',
        '--test_list',
        default='data/test_tfrecords_se.csv',
        help='test_list')

    argparser.add_argument(
        '-b',
        '--batchsize',
        default=100,
        type=int,
        help='Batch size for training and validation')

    argparser.add_argument(
        '-t',
        '--timesteps',
        default=500,
        type=int,
        help='rnn timesteps for training and validation')

    argparser.add_argument(
        '-q',
        '--quantized',
        default=False,
        type=bool,
        help='is post quantization?')

    argparser.add_argument(
        '-l',
        '--learning_rate',
        default = 4 * 10**-4,
        type=float,
        help='learning rate')

    argparser.add_argument(
        '-n',
        '--num_epoch',
        type=int,
        default=1000,
        help='Number of epochs to train')

    argparser.add_argument(
        '-e',
        '--epoch_loaded',
        default='random',
        help='epoch_loaded = \'random\': weight table is randomly generated, \
              epoch_loaded = \'latest\': weight table is loaded from the latest saved epoch result \
              epoch_loaded = 10  \
              (or any non-negative integer): weight table is loaded from epoch 10')

    main(argparser.parse_args())
