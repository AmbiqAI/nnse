"""
Training script for SE RNN
"""
import os
import re
import datetime
import logging
import argparse
import pickle
import tensorflow as tf
import numpy as np
import yaml
import matplotlib.pyplot as plt
from nnsp_pack.nn_module_new import NeuralNetClass
from nnsp_pack.statsClass import tf_round
from nnsp_pack.tfrecord_converter_se_split import tfrecords_pipeline
from nnsp_pack.loss_functions import loss_mse
from nnsp_pack.learn_rate import NoamAnnelingLRSchedule, CosineSchedule
from nnsp_pack.converter_fix_point import fakefix_tf
from nnsp_pack.calculate_feat_stats_se_split import feat_stats_estimator
from nnsp_pack.load_nn_arch import load_nn_arch, setup_nn_folder
from nnsp_pack.tf_basic_math import tf_log10_eps
# import c_code_table_converter

SHOW_STEPS          = False
DISPLAY_HISTOGRAM   = False
BLOCKS_PER_AUDIO    = 5
DIM_TARGET          = 257
PI                  = 3.1415926
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
        quantized   = False,
    ):
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
                amp_s,        # clean
                amp_sn * est, # noisy * mask
                mask,
                exp=0.6)

    if training:
        gradients = tape.gradient(ave_loss, net.trainable_variables)

        gradients_clips = [ grad
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
        feat_type       = 'mel',
        train_summary_writer=None,):
    """
    Training for one epoch
    """
    dim_feat, = norm_mean.shape

    net.stats_inst.reset_stats()
    states = net.make_states(batchsize, zero_state=zero_state)

    total_batches = int(len(fnames) / batchsize)

    for batch, data in enumerate(dataset):
        if num_context - 1 > 0:
            if batch % BLOCKS_PER_AUDIO == 0:
                # states = make_lstm_states(net, batchsize, zero_state=zero_state)

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
        if num_context - 1 > 0:
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
                    quantized   = quantized,
                 )

            _, states, ave_loss, steps = tmp

            net.stats_inst.update_cost_steps(ave_loss, steps)

        if batch % BLOCKS_PER_AUDIO == (BLOCKS_PER_AUDIO-1):
            tf.print(f"\r {int(batch / 5)}/{total_batches}: ",
                        end = '')

            net.stats_inst.show_loss(
                net.config[-1]['layer_neurons'],
                SHOW_STEPS,
                lr=optimizer.learning_rate)

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

def test(
        args,
        nn_train,
        config,
        stats,
        quantized):
    """ test function"""
    from nnsp_pack.feature_module import FeatureClass, display_stft_all
    from nnsp_pack.basic_dsp import dc_remove
    import soundfile as sf
    import librosa

    num_context = config['feat']['num_context']
    dim_feat = config['nn_arch'][0]['layer_neurons']
    feat_type = config['feat']['type']
    wavfile = args.test_wavefile
    # wavfile = 'test_wavs/steak_hairdryer.wav'

    audio, fs = sf.read(wavfile)
    if audio.ndim > 1:
        audio=audio[:,0]
    if fs > 16000:
        audio = librosa.resample(
                audio,
                orig_sr=fs,
                target_sr=16000)

    speech = dc_remove(audio)

    params_audio_def = {
        'win_size'      : 480,
        'hop'           : 160,
        'len_fft'       : 512,
        'sample_rate'   : 16000,
        'nfilters_mel'  : 72 }
    feat_inst      = FeatureClass(
                            win_size        = params_audio_def['win_size'],
                            hop             = params_audio_def['hop'],
                            len_fft         = params_audio_def['len_fft'],
                            sample_rate     = params_audio_def['sample_rate'],
                            nfilters_mel    = params_audio_def['nfilters_mel'])

    spec_sn, _, feat_sn, pspec_sn = feat_inst.block_proc(speech)
    
    pspec_sn_tmp = tf.constant(pspec_sn, dtype=tf.float32)
    pspec_sn_tmp = tf.expand_dims(pspec_sn_tmp, 0)
    if feat_type == 'mel':
            feats = tf.matmul(pspec_sn_tmp, MEL_FBANKS)
    elif feat_type == 'pspec':
        feats = tf.identity(pspec_sn_tmp)
    feats = tf_log10_eps(feats)
    feats = fakefix_tf(feats, 32, 15)
    if num_context - 1 > 0:
        shape = (1, num_context-1, dim_feat)

        padddings_tsteps = tf.constant(
                        np.full(shape, np.log10(2**-15)),
                        dtype = tf.float32)
        feats = tf.concat([padddings_tsteps, feats], 1)
    nfeats = (feats - stats['nMean_feat']) * stats['nInvStd']
    nfeats = fakefix_tf(nfeats, 16, 8)
    states = nn_train.make_states(1, zero_state=True)

    est, states= nn_train(
        nfeats,mask=1.0,
        states=states,
        training=False,
        quantized=quantized)

    tfmask=est[0].numpy()
    nfeats=nfeats[0].numpy()[num_context-1:,:]
    feats=feats[0].numpy()[num_context-1:,:]

    audio_out = feat_inst.istft_frame_proc(
        spec_sn,
        tfmask
    )
    name_model = re.sub(r'\.yaml', '', os.path.basename(args.config_file))
    name= re.sub(r'\.wav', '', f'{os.path.basename(wavfile)}')

    folder=f'test_results/{name_model}/{name}'
    os.makedirs(folder, exist_ok=True)
    sf.write(f'{folder}/noisy.wav', speech, 16000)
    sf.write(f'{folder}/enhance.wav', audio_out, 16000)

    print(f'Check your noisy speech in test_results/{name}/noisy.wav')
    print(f'Check your enhanced speeech in test_results/{name}/enhance.wav')

    plt.figure(1)
    plt.clf()

    plt.subplot(3,1,1)
    plt.imshow(
        nfeats.T,
        origin      = 'lower',
        cmap        = 'pink_r',
        aspect      = 'auto')
    plt.subplot(3,1,2)
    plt.imshow(
        feats.T,
        origin      = 'lower',
        cmap        = 'pink_r',
        aspect      = 'auto')
    plt.subplot(3,1,3)
    plt.imshow(
        tfmask.T,
        origin      = 'lower',
        cmap        = 'pink_r',
        aspect      = 'auto')
    plt.savefig(f'{folder}/feat_mask.pdf')

def make_folder(config_file):
    """ make folder"""
    name_model = re.sub(r'\.yaml', '', os.path.basename(config_file))
    name_model = re.sub(r'config_', '', name_model)
    os.makedirs(f'models_trained', exist_ok=True)
    os.makedirs(f'models_trained/{name_model}', exist_ok=True)
    folder_nn = f'models_trained/{name_model}'
    return folder_nn

def main(args):
    """
    main function to train neural network training
    """
    batchsize       = args.batchsize
    timesteps       = args.timesteps
    num_epoch       = args.num_epoch
    epoch_loaded    = args.epoch_loaded
    quantized       = args.quantized
    if args.mode == 'test':
        batchsize = 1
    tfrecord_list = {   'train' : args.train_list,
                        'test'  : args.test_list}
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config_nn = config['nn_arch']
        config_feat = config['feat']

    folder_nn = make_folder(args.config_file)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'tensorboard/{folder_nn}/logs/{current_time}'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    dim_feat = config_nn[0]['layer_neurons']

    nn_train = NeuralNetClass(
        config=config_nn,
        batchsize   = batchsize)

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

        print(f"(train) best epoch picked by loss = {np.argmin(loss['train'][0: epoch_loaded+1])}")
        print(f"(test)  best epoch picked by loss = {np.argmin(loss['test'][0: epoch_loaded+1])}")

    fnames = {}
    for tr_set in ['train', 'test']:
        with open(tfrecord_list[tr_set], 'r') as file: # pylint: disable=unspecified-encoding
            try:
                lines = file.readlines()
            except:# pylint: disable=bare-except
                print(f'Can not find the list {tfrecord_list[tr_set]}')
            else:
                len0 = int(len(lines) / batchsize) * batchsize
                fnames[tr_set] = [line.strip() for line in lines[:len0]]

                fnames[tr_set] = filter_in_data(fnames[tr_set])
                # fnames[tr_set] = fnames[tr_set][1:100] 
    shift_step = int(np.ceil(len(fnames['train']) / batchsize))
    lr_schedule=CosineSchedule(
    base_lr=args.learning_rate,
    warmup_steps=2000,
    total_steps=shift_step*num_epoch,
    start_step=(epoch1_loaded * shift_step))

    optimizer = tf.keras.optimizers.Adam(
        # learning_rate=args.learning_rate
        learning_rate=lr_schedule,
        weight_decay=0.1
        )
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
                feat_type=config['feat']['type'])

    # nn_np = c_code_table_converter.tf2np(nn_train, quantized=quantized)
    # if DISPLAY_HISTOGRAM:
    #     c_code_table_converter.draw_nn_hist(nn_np)
    #     c_code_table_converter.draw_nn_weight(
    #         nn_np,
    #         nn_train,
    #         pruning=False)

    tot=0
    for v in nn_train.trainable_variables:
        tot+=v.numpy().size
    print(f"Total number of parameters: {tot}")

    if args.mode == 'test':
        test(args, nn_train, config,  stats, quantized)
        return

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
                num_dnsampl     = 1,
                num_context     = config_feat['num_context'],
                quantized       = quantized,
                feat_type       = config_feat['type'],
                )

        # Computing Training loss
        epoch_proc(
            nn_train,
            optimizer,
            dataset_tr,
            fnames['train'],
            batchsize,
            timesteps,
            training        = False,
            zero_state      = True,
            norm_mean       = stats['nMean_feat'],
            norm_inv_std    = stats['nInvStd'],
            num_dnsampl     = 1,
            num_context     = config_feat['num_context'],
            quantized       = quantized,
            feat_type       = config_feat['type'],)

        loss['train'][epoch] = nn_train.stats_inst.stats['acc_loss'] / nn_train.stats_inst.stats['acc_steps']
        loss['train'][epoch] /= config_nn[-1]['layer_neurons']

        acc['train'][epoch] = nn_train.stats_inst.stats['acc_matchCount'] / nn_train.stats_inst.stats['acc_steps']

        # Computing Testing loss
        epoch_proc(
            nn_train,
            optimizer,
            dataset_te,
            fnames['test'],
            batchsize,
            timesteps,
            training            = False,
            zero_state          = True,
            norm_mean           = stats['nMean_feat'],
            norm_inv_std        = stats['nInvStd'],
            num_dnsampl         = 1,
            num_context         = config_feat['num_context'],
            quantized           = quantized,
            feat_type           = config_feat['type'],)

        loss['test'][epoch] = nn_train.stats_inst.stats['acc_loss'] / nn_train.stats_inst.stats['acc_steps']
        loss['test'][epoch] /= config_nn[-1]['layer_neurons']

        acc['test'][epoch] = nn_train.stats_inst.stats['acc_matchCount'] / nn_train.stats_inst.stats['acc_steps']
        nn_train.save_weights(f'{folder_nn}/checkpoints/model_checkpoint_ep{epoch}')

        fr = f'{folder_nn}/loss_acc.png'
        plt.savefig(fr)

        with open(os.path.join(folder_nn, 'nn_loss.pkl'), "wb") as file:
            pickle.dump(loss, file)
        with open(os.path.join(folder_nn, 'nn_acc.pkl'), "wb") as file:
            pickle.dump(acc, file)

        tf.print('Epoch spent ', tf_round(tf.timestamp() - t_start), ' seconds')
        print(f"(train) best epoch picked by loss = {np.argmin(loss['train'][0: epoch+1])}")
        print(f"(test)  best epoch picked by loss = {np.argmin(loss['test'][0: epoch+1])}")

        # tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('loss/train', loss['train'][epoch], step=epoch)
            tf.summary.scalar('loss/test', loss['test'][epoch], step=epoch)

def filter_in_data(fnames):
    """_summary_

    Args:
        fnames (_type_): _description_

    Returns:
        _type_: _description_
    """
    ntypes = [
            'ESC-50-MASTER',
            'wham_noise',
            'FSD50K',
            'musan',
        ]
    fnames_out=[]
    for fname in fnames:
        for ntype in ntypes:
            if re.search(fr"{ntype}", fname):
                fnames_out+=[fname]
                break
    return fnames_out

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(
        description='Training script for se model')

    argparser.add_argument(
        '-m',
        '--mode',
        default='train',
        help='test or train')

    argparser.add_argument(
        '-tw',
        '--test_wavefile',
        default='test_wavs/steak_hairdryer.wav',
        help='test_wavs')

    argparser.add_argument(
        '-a',
        '--config_file',
        default='nn_arch/config_unet_relu_large.yaml',
        help='nn architecture')

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
        default =  4 * 10**-4,
        type=float,
        help='learning rate')

    argparser.add_argument(
        '-n',
        '--num_epoch',
        type=int,
        default=150,
        help='Number of epochs to train')

    argparser.add_argument(
        '-e',
        '--epoch_loaded',
        default="random",
        help='epoch_loaded = \'random\': weight table is randomly generated, \
              epoch_loaded = \'latest\': weight table is loaded from the latest saved epoch result \
              epoch_loaded = 10  \
              (or any non-negative integer): weight table is loaded from epoch 10')

    main(argparser.parse_args())
