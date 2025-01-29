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
RESET_EVERY_AUDIO_CLIP = False
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
                states,
                mask=mask,
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
        feat_stats,
        config_feat,
        epoch = 0,
        quantized       = False,
        train_summary_writer=None,):
    """
    Training for one epoch
    """
    norm_mean       = feat_stats['nMean_feat']
    norm_inv_std    = feat_stats['nInvStd']
    num_lookahead   = config_feat['num_lookahead']
    feat_type = config_feat['type']
    net.stats_inst.reset_stats()

    def reset_states():
        """
        Reset states
        """
        states = net.make_states(
            batchsize,
            norm_mean=norm_mean,
            norm_inv_std = norm_inv_std,
            zero_state=zero_state)
        if num_lookahead > 0: # non-causal
            shape = (batchsize, num_lookahead, net.dim_trgt)
            head_s = tf.zeros(shape, dtype = tf.float32)
            head_sn = tf.zeros(shape, dtype = tf.float32)
        return states, head_s, head_sn

    total_batches =  len(fnames) // batchsize

    # intialize the padding for the first batch
    states, head_pspec_s, head_pspec_sn = reset_states()

    total_steps=BLOCKS_PER_AUDIO * total_batches * epoch
    for batch, data in enumerate(dataset):
        if train_summary_writer is not None:
            with train_summary_writer.as_default():
                tf.summary.scalar(
                    'learning_rate',
                    optimizer.learning_rate,
                    step=total_steps)
                tf.summary.scalar(
                    'steps',
                    total_steps,
                    step=epoch)
            total_steps+=1
        pspec_sn, masks, pspec_s, _ = data

        if feat_type == 'mel':
            feats = tf.matmul(pspec_sn, MEL_FBANKS)
        elif feat_type == 'pspec':
            feats = tf.identity(pspec_sn)
        feats = tf_log10_eps(feats)
        feats = fakefix_tf(feats, 32, 15)
        nfeats = (feats - norm_mean) * norm_inv_std
        nfeats = fakefix_tf(nfeats, 16, 8)

        if RESET_EVERY_AUDIO_CLIP:
            if batch % BLOCKS_PER_AUDIO == (BLOCKS_PER_AUDIO-1):
                states, head_pspec_s, head_pspec_sn = reset_states()
        if num_lookahead > 0: # non-causal
            tmp_s = tf.concat([head_pspec_s, pspec_s], 1)
            tmp_sn = tf.concat([head_pspec_sn, pspec_sn], 1)
            head_pspec_s  =  tf.identity(pspec_s[:,-num_lookahead:,:])
            head_pspec_sn = tf.identity(pspec_sn[:,-num_lookahead:,:])
            pspec_s  =  tmp_s[:,:-num_lookahead,:]
            pspec_sn = tmp_sn[:,:-num_lookahead,:]

        _, steps, _ = pspec_sn.shape
        for k in range(steps // timesteps):

            start = k * timesteps
            end = (k+1) * timesteps
            tmp = train_kernel(
                    tf.identity(  nfeats[:,start:end,:]),
                    tf.identity(pspec_sn[:,start:end,:]),
                    tf.identity( pspec_s[:,start:end,:]),
                    tf.identity(   masks[:,start:end,:]),
                    states,
                    net,
                    optimizer,
                    training    = training,
                    quantized   = quantized,
                 )

            _, states, ave_loss, steps = tmp

            net.stats_inst.update_cost_steps(ave_loss, steps)

        if batch % BLOCKS_PER_AUDIO == (BLOCKS_PER_AUDIO-1):
            tf.print(f"\r {batch // BLOCKS_PER_AUDIO}/{total_batches}: ",
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
        feat_stats,
        quantized):
    """ test function"""
    from nnsp_pack.feature_module import FeatureClass, display_stft_all
    from nnsp_pack.basic_dsp import dc_remove
    from nnsp_pack.tflite_convert import warp_tf_model, tflite_convert
    import soundfile as sf
    import librosa

    num_lookahead = config['feat']['num_lookahead']
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

    nfeats = (feats - feat_stats['nMean_feat']) * feat_stats['nInvStd']
    nfeats = fakefix_tf(nfeats, 16, 8)
    states = nn_train.make_states(
        batchsize=1,
        norm_mean=feat_stats['nMean_feat'],
        norm_inv_std=feat_stats['nInvStd'],
        zero_state=True)
    stream=True

    if stream: # frame by frame processing for streaming application
        nn_train = warp_tf_model(nn_train, time_steps=1)
        est = []
        for i in range(nfeats.shape[1]):
            print(f"\rProcessing frame {i}/{nfeats.shape[1]}", end = '')
            est0, states = nn_train(inputs=[nfeats[:,i:i+1,:], states])
            est += [est0]
        est = tf.concat(est, 1)
    else:
        nn_train = warp_tf_model(nn_train, time_steps=nfeats.shape[1])
        est, states = nn_train(
            [nfeats, states],
            )
    nn_train.summary()
    print(nfeats.shape)
    print(nfeats.numpy().min(), nfeats.numpy().max())
    for state in states[0][0]:
        print(state.shape)
        print(state.numpy().min(), state.numpy().max())
    for state in states[0][1]:
        print(state.shape)
        print(state.numpy().min(), state.numpy().max())

    tflite_fp16_model = tflite_convert(
        nn_train, nbit=8, path_tflite='./tflite/nnse_int8.tflite')
    # interpreter = tf.lite.Interpreter(model_content=tflite_fp16_model)
    # interpreter.allocate_tensors()  # Needed before execution!
    
    # input = interpreter.get_input_details()[0] 
    # import pdb; pdb.set_trace()

    tfmask=est[0].numpy()
    nfeats=nfeats[0].numpy()[num_lookahead:,:]
    feats=feats[0].numpy()[num_lookahead:,:]

    if num_lookahead > 0: # non-causal
        tfmask = tfmask[num_lookahead:,:]
        spec_sn = spec_sn[:-num_lookahead,:]

    audio_out, spec_en = feat_inst.istft_frame_proc(
        spec_sn,
        tfmask
    )

    pspec_en = np.log10(np.abs(spec_en)+10**-5)
    pspec_sn = np.log10(np.abs(spec_sn)+10**-5)

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
        pspec_en.T,
        origin      = 'lower',
        cmap        = 'pink_r',
        aspect      = 'auto')
    plt.title('Enhanced')
    plt.colorbar()

    plt.subplot(3,1,2)
    plt.imshow(
        pspec_sn.T,
        origin      = 'lower',
        cmap        = 'pink_r',
        aspect      = 'auto')
    plt.title('Noisy')
    plt.colorbar()

    plt.subplot(3,1,3)
    plt.imshow(
        tfmask.T,
        origin      = 'lower',
        cmap        = 'pink_r',
        aspect      = 'auto',
        vmin=0,
        vmax=1)
    plt.title('TF-Mask')
    plt.colorbar()

    plt.savefig(f'{folder}/feat_mask.pdf')
    plt.show()

def make_savedModel_folder(config_file):
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

    folder_nn = make_savedModel_folder(args.config_file)

    if args.mode == 'train':
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
            epoch_loaded=int(epoch_loaded)
            nn_train.load_weights(
                f'{folder_nn}/checkpoints/model_checkpoint_ep{int(epoch_loaded)}')
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
    shift_step = BLOCKS_PER_AUDIO * (len(fnames['train']) // batchsize)

    lr_schedule=CosineSchedule(
        base_lr=args.learning_rate,
        warmup_steps=2000,
        total_steps=num_epoch * shift_step,
        start_step =epoch1_loaded * shift_step)

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
            feat_stats = pickle.load(file)
    else:
        feat_stats = feat_stats_estimator(
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
        test(args, nn_train, config, feat_stats, quantized)
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
                feat_stats      = feat_stats,
                config_feat     = config_feat,
                epoch           = epoch,
                quantized       = quantized,
                train_summary_writer=train_summary_writer,
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
            feat_stats      = feat_stats,
            config_feat     = config_feat,
            epoch           = epoch,
            quantized       = quantized,
            )

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
            feat_stats          = feat_stats,
            config_feat         = config_feat,
            epoch               = epoch,
            quantized           = quantized,
            )

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
        type=str,
        help='test or train')

    argparser.add_argument(
        '-tw',
        '--test_wavefile',
        default='test_wavs/keyboard_steak.wav',
        help='test_wavs')

    argparser.add_argument(
        '-a',
        '--config_file',
        default='nn_arch/config_unet_relu_large_noncausal_sep.yaml',
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
        default="latest",
        help='epoch_loaded = \'random\': weight table is randomly generated, \
              epoch_loaded = \'latest\': weight table is loaded from the latest saved epoch result \
              epoch_loaded = 10  \
              (or any non-negative integer): weight table is loaded from epoch 10')

    main(argparser.parse_args())
