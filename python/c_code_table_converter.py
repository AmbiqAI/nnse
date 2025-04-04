"""
Convert the tensorflow NN weight table to C code
"""
import os
import pickle
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from nnsp_pack import c_weight_man
from nnsp_pack.nn_module_new import NeuralNetClass
from nnsp_pack.load_nn_arch import load_nn_arch, setup_nn_folder
from mpl_toolkits.axes_grid1 import make_axes_locatable
from data_se import params_audio as PARAMS_AUDIO
# PARAMS_AUDIO = {
#     'win_size'      : 480,
#     'hop'           : 160,
#     'len_fft'       : 512,
#     'sample_rate'   : 16000,
#     'nfilters_mel'  : 40
# }
def float2fix(data_in, nfrac, bitwidth):
    """
    Floating point to int
    """
    max_val = 2**(bitwidth-1) - 1
    min_val = -2**(bitwidth-1)

    out = np.minimum(np.maximum(np.floor(data_in * 2**nfrac), min_val), max_val).astype(int)

    return out

def fix2hex(data_in, nbit):
    """
    integer point to hex
    """
    if data_in < 0:
        data_in = 2**nbit + data_in
    return data_in

def tf2np(net_tf, quantized = False):
    """
    Convert tensor to np array
    """
    if quantized:
        net_tf.quantized_weight()

    net_np = []
    layer_types = net_tf.get_config_info('layer_type')
    neurons = net_tf.get_config_info('layer_neurons')
    for i, layer in enumerate(net_tf.nn_layers):
        size_conv1d = 1
        nbits_w = net_tf.bitwidths['kernel']
        qbits_w = net_tf.nfracs['kernel'][i].numpy()

        nbits_b = net_tf.bitwidths['bias']
        qbits_b = net_tf.nfracs['bias'][i].numpy()

        layer = layer.trainable_variables
        nn_type = layer_types[i]
        for val in layer:
            if re.search(r'/kernel', val.name):
                kernel_f = val.numpy()

            elif re.search(r'/recurrent_kernel', val.name):
                kernel_r = val.numpy()

            elif re.search(r'/bias', val.name):
                bias = val.numpy()

        if nn_type == 'lstm':
            neuron = neurons[i+1]

            if len(bias) == neuron * 8: # for old tf version lstm
                tmp1, tmp2 = np.split(bias, 2) # pylint: disable=unbalanced-tuple-unpacking
                bias = tmp1 + tmp2
            # kernel
            kernel = np.concatenate((kernel_f, kernel_r), axis = 0)
            i_state, f_state, j_state, o_state = np.split(kernel, 4, axis = 1) # pylint: disable=unbalanced-tuple-unpacking
            kernel = np.concatenate((i_state, j_state, f_state, o_state), axis = 1)
            # bias
            i_state, f_state, j_state, o_state = np.split(bias, 4) # pylint: disable=unbalanced-tuple-unpacking
            bias = np.concatenate((i_state, j_state, f_state, o_state))

        else:
            if nn_type == 'fc':
                kernel = kernel_f

            elif nn_type == 'conv1d':
                shape = kernel_f.shape # (6, dim_feat, 1, neurons)
                size_conv1d = shape[0]
                kernel = np.transpose(kernel_f[:,:,0,:], (2,0,1))
                kernel = np.reshape(kernel, (kernel.shape[0], -1))
                kernel = kernel.T

        net_np += [{ 'kernel'       : kernel,
                    'bias'          : bias,
                    'nbits_b'       : nbits_b,
                    'qbits_b'       : qbits_b,
                    'nbits_w'       : nbits_w,
                    'qbits_w'       : qbits_w,
                    'size_conv1d'   : size_conv1d}]
    return net_np

def draw_nn_hist(nn_table):
    """
    Display NN weight table histogram
    """
    plt.title('kernels and biases histogram. Close the figure to continue')
    for i, layer in enumerate(nn_table):
        ax_h = plt.subplot(2, len(nn_table), i+1)
        ax_h.grid(False)
        ax_h.set_axis_off()
        data = layer['kernel'].flatten()
        ax_h.hist(data, bins=300)

        ax_h = plt.subplot(2, len(nn_table), i + len(nn_table) +1)
        ax_h.grid(False)
        ax_h.set_axis_off()
        data = layer['bias'].flatten()
        ax_h.hist(data, bins=1000)

    plt.show()

def draw_nn_weight(nn_table, nn_infer, pruning=False, hard_thresh = None):
    """
    Display NN weight table histogram
    """
    plt.title('kernels and biases histogram. Close the figure to continue')
    fig = plt.figure(2)
    for i, layer in enumerate(nn_table):
        ax_h = plt.subplot(3, len(nn_table), i+1)
        data = layer['kernel']
        data = np.abs(data)
        if pruning:
            try:
                reg, mask = nn_infer.nn_layers[i].get_prune_reg_mask()
            except:  # pylint: disable=W0702
                mask = 1.0
            else:
                if hard_thresh:
                    mask = (reg.numpy() > hard_thresh).astype(np.float32)
                else:
                    mask = np.array([1,])
                print(f"Sparsity {mask.sum().astype(np.int32)}/{mask.size}")

                if nn_infer.layer_types[i]=='lstm':
                    mask = np.tile(mask,4)
        img = ax_h.imshow(
            data,
            vmin=data.min(),
            vmax=data.max(),
            cmap='pink',
            aspect='auto')
        divider = make_axes_locatable(ax_h)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img, cax=cax, orientation='vertical')

        if pruning:
            ax_h = plt.subplot(3, len(nn_table), i+1+len(nn_table))
            if 0:
                prune_data = layer['kernel'] * mask
                prune_data = np.abs(prune_data)
                img = ax_h.imshow(
                    prune_data,
                    vmin=prune_data.min(),
                    vmax=prune_data.max(),
                    cmap='pink',
                    aspect='auto')
                divider = make_axes_locatable(ax_h)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(img, cax=cax, orientation='vertical')
            else:
                plt.plot(np.sort(reg)[::-1])

        ax_h = plt.subplot(3, len(nn_table), i+1+len(nn_table) * 2)
        data = data.flatten()
        plt.plot(np.sort(data))

    plt.show()

def converter(  net_tf,
                stats,
                nn_id = 0,
                nn_name = 'nn_model',
                make_c_table = True,
                folder_c = ".",
                arm_core = 'M4',
                num_dnsampl=1,
                is_tflite=1,
                tflite_filename='nnse_int16.tflite'):
    """
    Convert tensor in NN to c code
    """

    mean, inv_std   = stats.values()
    activations     = net_tf.get_config_info('activation')
    layer_types     = net_tf.get_config_info('layer_type')
    neurons         = net_tf.get_config_info('layer_neurons')

    if is_tflite==1:
        from nnsp_pack.tflite_convert import warp_tf_model, tflite_convert
        nn = warp_tf_model(net_tf, dim_feat=neurons[0])
        tflite_convert(
            nn,
            dtype='int16',
            path_tflite=f'{folder_c}/{tflite_filename}')
        net_np=None
    else:
        net_tf.quantized_weight()
        net_np = tf2np(net_tf)
    if make_c_table:
        # fname_inc = f'../evb/src/def_nn{nn_id}_{nn_name}.h'
        fname_inc = f'{folder_c}/def_nn{nn_id}_{nn_name}.h'
        with open(fname_inc, 'w') as file: # pylint: disable=unspecified-encoding
            file.write(f'#ifndef __DEF_NN{nn_id}_{nn_name.upper()}__\n')
            file.write(f'#define __DEF_NN{nn_id}_{nn_name.upper()}__\n')
            file.write('#include <stdint.h>\n')
            file.write('#include "neural_nets.h"\n')
            file.write('#include "nn_speech.h"\n')
            file.write(f'extern const int32_t feature_mean_{nn_name}[];\n')
            file.write(f'extern const int32_t feature_stdR_{nn_name}[];\n')
            if is_tflite==0: # for nnsp
                file.write(f'extern NeuralNetClass net_{nn_name};\n' )
            file.write(f'extern PARAMS_NNSP params_nn{nn_id}_{nn_name};\n' )
            file.write('#endif\n')

        #--------------Header-----------------#
        # fname_c = f'../evb/src/def_nn{nn_id}_{nn_name}.c'
        fname_c = f'{folder_c}/def_nn{nn_id}_{nn_name}.c'
        with open(fname_c, 'w') as file: # pylint: disable=unspecified-encoding
            file.write(f"// arm_core = cortex-{arm_core}\n")
            file.write('#include <stdint.h>\n')
            file.write('#include "neural_nets.h"\n')
            file.write('#include "activation.h"\n')
            file.write('#include "affine.h"\n')
            file.write('#include "lstm.h"\n')
            file.write('#include "nn_speech.h"\n')
            file.write(f"extern const int16_t stft_win_coeff_w{PARAMS_AUDIO['win_size']}_h{PARAMS_AUDIO['hop']}[];\n") # pylint: disable=line-too-long
            file.write(f"PARAMS_NNSP params_nn{nn_id}_{nn_name} = {{\n")
            file.write(f"\t.samplingRate = {PARAMS_AUDIO['sample_rate']},\n")
            file.write(f"\t.fftsize = {PARAMS_AUDIO['len_fft']},\n")
            file.write(f"\t.winsize_stft = {PARAMS_AUDIO['win_size']},\n")
            file.write(f"\t.hopsize_stft = {PARAMS_AUDIO['hop']},\n")
            file.write(f"\t.num_mfltrBank = {neurons[0]},\n")
            file.write(f"\t.num_dnsmpl = {num_dnsampl},\n")
            file.write(f"\t.pt_stft_win_coeff = stft_win_coeff_w{PARAMS_AUDIO['win_size']}_h{PARAMS_AUDIO['hop']},\n") # pylint: disable=line-too-long
            file.write("\t.start_bin = 0,\n")
            file.write("\t.is_dcrm = 1,\n")
            file.write("\t.pre_gain_q1 = 1 << 1, // q1\n")
            file.write('};\n')

            #-----------------stats---------------------------------
            file.write('/*************stats***********/\n')
            file.write(f'const int32_t feature_mean_{nn_name}[] = {{')
            for val in mean:
                tmp = int(val * 2**15)
                file.write(f'0x{fix2hex(tmp, nbit=32):08x}, ')
            file.write('};\n')

            file.write(f'const int32_t feature_stdR_{nn_name}[] = {{')
            for val in inv_std:
                tmp = int(val * 2**15)
                file.write(f'0x{fix2hex(tmp, nbit=32):08x}, ')
            file.write('};\n')

            if is_tflite==0: # for nnsp
                total_bytes = 0
                #-----------------weight table---------------------------------
                for i, layer_type in enumerate(layer_types):
                    file.write(f'// layer {i}\n')
                    
                    if layer_type in ('fc', 'conv1d'):
                        kernel = net_np[i]['kernel'].T
                        qbit = net_np[i]['qbits_w']
                        
                        kernel = c_weight_man.c_matrix_man(kernel, arm_core)
                        
                        kernel = float2fix(kernel, qbit, 8)
                        
                        file.write(f'uint8_t {nn_name}_kernel{i}[]={{')
                        for k in kernel:
                            file.write(f'0x{fix2hex(k, nbit=8):02x},' )
                        file.write('};\n')
                        total_bytes += len(kernel)

                        bias = net_np[i]['bias'].T
                        qbit = net_np[i]['qbits_b']

                        bias = float2fix(bias, qbit, 16)
                        file.write(f'uint16_t {nn_name}_bias{i}[]={{')
                        for item_bias in bias:
                            file.write(f'0x{fix2hex(item_bias, nbit=16):04x},')
                        file.write('};\n')
                        total_bytes += (len(bias) * 2)

                    elif layer_type=='lstm':
                        kernel = net_np[i]['kernel'].T
                        kernel_f, kernel_r = np.split(kernel, 2, axis=1) # pylint: disable=unbalanced-tuple-unpacking
                        
                        bias = net_np[i]['bias'].T
                        kernel_f, kernel_r, bias = c_weight_man.c_lstm_weight_man(
                                                    kernel_f, kernel_r, bias, arm_core)

                        qbit = net_np[i]['qbits_w']
                        kernel_f = float2fix(kernel_f, qbit, 8)
                        file.write(f'uint8_t {nn_name}_kernel{i}[]={{')
                        for k in kernel_f:
                            file.write(f'0x{fix2hex(k, nbit=8):02x},' )
                        file.write('};\n')
                        total_bytes += len(kernel_f)

                        kernel_r = float2fix(kernel_r, qbit, 8)
                        file.write(f'uint8_t {nn_name}_kernel_rec{i}[]={{')
                        for k in kernel_r:
                            file.write(f'0x{fix2hex(k, nbit=8):02x},')
                        file.write('};\n')
                        total_bytes += len(kernel_r)

                        qbit = net_np[i]['qbits_b']
                        bias = float2fix(bias, qbit, 16)
                        file.write(f'uint16_t {nn_name}_bias{i}[]={{')
                        for item_b in bias:
                            file.write(f'0x{fix2hex(item_b, nbit=16):04x},')
                        file.write('};\n')
                        total_bytes += (len(bias) * 2)
                print(f"total size = {total_bytes} bytes")

                #-----------------nn struct ---------------------------------
                file.write('// lstm states\n')
                total_layers_lstm = 0
                for i, layer_type in enumerate(layer_types):
                    if layer_type == 'lstm':
                        file.write(f'int32_t cstate_layer{i}_{nn_name}[{neurons[i+1]}];\n')
                        file.write(f'int16_t hstate_layer{i}_{nn_name}[{neurons[i+1]}];\n')
                        total_layers_lstm += 1

                file.write(f'NeuralNetClass net_{nn_name} = {{\n\n' )
                file.write(f'\t{len(net_np)}, // layers\n\n')

                file.write('\t{')
                for i, neuron in enumerate(neurons[:-1]):
                    if layer_types[i]=='conv1d':
                        file.write(f'{neuron * net_np[i]["size_conv1d"]},')
                    else:
                        file.write(f'{neuron},')
                file.write(f'{neurons[-1]},' )
                file.write('}, // nn size for each layer, including the input layer\n\n')

                file.write('\t{')
                for layer_type in layer_types:
                    if layer_type=='conv1d':
                        file.write('fc,')
                    else:
                        file.write(f'{layer_type},' )
                file.write('}, // layer type\n\n')

                file.write('\t{')
                for layer in net_np:
                    file.write(f'{int(layer["qbits_w"]):d},')
                file.write('}, // fractional bits (kernel)\n\n')

                file.write('\t{')
                file.write('8,') # layer 0
                for act in activations[:-1]:
                    if act=='tanh':
                        file.write('15,')
                    elif act=='sigmoid':
                        file.write('15,')
                    elif act=='relu6':
                        file.write('12,')
                    elif act=='linear':
                        file.write('15,')
                file.write('}, // qbit_i\n\n')

                file.write('\t{')
                for layer in net_np:
                    file.write(f'{int(layer["qbits_b"]):d},')
                file.write('}, // fractional bits (bias)\n\n')

                file.write('\t{')
                for act in activations:
                    if act=='tanh':
                        file.write('ftanh,')
                    elif act=='sigmoid':
                        file.write('sigmoid,')
                    else:
                        file.write(f'{act},')
                file.write('}, // activations\n\n')

                cnt = 0
                file.write('\t{\n')
                for i, layer_type in enumerate(layer_types):
                    if layer_type in ('fc', 'conv1d'):
                        file.write('\t\t(int32_t*) 0,\n')
                    elif layer_type=='lstm':
                        file.write(f'\t\t(int32_t*) cstate_layer{i}_{nn_name},\n')
                        # file.write(f'\t\t(int32_t*) lstm_{nn_name}[{cnt}].c_states,\n')
                        cnt+=1
                file.write('\t}, // cstates lstm\n\n')

                cnt = 0
                file.write('\t{\n')
                for i, layer_type in enumerate(layer_types):
                    if layer_type in ('fc', 'conv1d'):
                        file.write('\t\t(int16_t*) 0,\n')
                    elif layer_type=='lstm':
                        # file.write(f'\t\t(int16_t*) lstm_{nn_name}[{cnt}].h_states,\n')
                        file.write(f'\t\t(int16_t*) hstate_layer{i}_{nn_name},\n')
                        cnt+=1
                file.write('\t}, // hstates lstm\n\n')

                file.write('\t{\n')
                for act in activations:
                    if act=='tanh':
                        file.write('\t\t(void* (*)(void*, int32_t*, int)) &tanh_fix,\n')
                    elif act=='sigmoid':
                        file.write('\t\t(void* (*)(void*, int32_t*, int)) &sigmoid_fix,\n')
                    elif act=='relu6':
                        file.write('\t\t(void* (*)(void*, int32_t*, int)) &relu6_fix,\n')
                    elif act=='linear':
                        file.write('\t\t(void* (*)(void*, int32_t*, int)) &linear_fix,\n')
                file.write('\t}, // activation function\n\n')

                file.write('\t{\n')
                for layer_type in layer_types:
                    if layer_type=='fc':
                        file.write('\t\t(int* (*)()) &fc_8x16,\n')
                    elif layer_type=='lstm':
                        file.write('\t\t(int* (*)()) &lstm_8x16,\n')
                    elif layer_type=='conv1d':
                        file.write('\t\t(int* (*)()) &fc_8x16,\n')
                file.write('\t}, // net layer type\n\n')

                file.write('\t{\n')
                for i, layer_type in enumerate(layer_types):
                    file.write(f'\t\t(int8_t*) {nn_name}_kernel{i},\n')
                file.write('\t}, // kernel\n\n')

                file.write('\t{\n')
                for i, layer_type in enumerate(layer_types):
                    file.write(f'\t\t(int16_t*) {nn_name}_bias{i},\n')
                file.write('\t}, // bias\n\n')

                file.write('\t{\n')
                for i, layer_type in enumerate(layer_types):
                    if layer_type=='lstm':
                        file.write(f'\t\t(int8_t*) {nn_name}_kernel_rec{i},\n')
                    else:
                        file.write('\t\t(int8_t*) 0,\n')
                file.write('\t}, // kernel_rec\n\n')

                file.write('};\n\n')

    return net_np, fname_inc, fname_c

def main(args):
    """
    main function to convert tensorflow Neural net model to c table
    """
    epoch_loaded    = int(args.epoch_loaded)
    config_file     = args.config_file
    nn_name         = args.net_name
    nn_id           = int(args.net_id)
    folder_c        = args.folder_c
    arm_core        = args.arm_core
    is_tflite       = args.is_tflite
    tflite_filename = args.tflite_filename
    import yaml
    from train_se import make_savedModel_folder
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config_nn = config['nn_arch']
        config_feat = config['feat']
    folder_nn = make_savedModel_folder(config_file)

    with open(os.path.join(f"{folder_nn}",'stats.pkl'), "rb") as file:
        feat_stats = pickle.load(file)
    nn_train = NeuralNetClass(
        config=config_nn,
        batchsize= 100,
        time_steps= 500,
        unroll_rnn=True,
        norm_mean=feat_stats['nMean_feat'],
        norm_inv_std=feat_stats['nInvStd'],
        )

    nn_train.load_weights(
            f'{folder_nn}/checkpoints/model_checkpoint_ep{epoch_loaded}' )
    nn_infer = NeuralNetClass(
        config=config_nn,
        batchsize= 1,
        time_steps= 1,
        unroll_rnn=True,
        norm_mean=feat_stats['nMean_feat'],
        norm_inv_std=feat_stats['nInvStd'],
        )

    # only copy the trainable variables
    for u,v in zip(nn_train.trainable_variables, nn_infer.trainable_variables):
        v.assign(u)
    # nn_infer.quantized_weight(batch_size=1)

    _, fname_inc, fname_c = converter(
            nn_infer,
            feat_stats,
            nn_name = nn_name,
            nn_id   = nn_id,
            folder_c= folder_c,
            arm_core  = arm_core,
            num_dnsampl=1,
            is_tflite=is_tflite,
            tflite_filename=tflite_filename
            )

    print(f'\nweight table is generated in \n{fname_inc}\n{fname_c}')

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='Convert trained Tensorflow model to C table')

    argparser.add_argument(
        '-a',
        '--config_file',
        default='nn_arch/config_unet_relu_noncausal_sep_specmel_th50.yaml',
        help='nn architecture')

    argparser.add_argument(
        '--epoch_loaded',
        default= 117,
        help='starting epoch')
    
    argparser.add_argument(
        '--is_tflite',
        type=int,
        default= 1,
        help='if true, using tflmodel, otherwise using nnsp model. ')

    argparser.add_argument(
        '--tflite_filename',
        type=str,
        default= 'nnse_int16.tflite',
        help='tflite_filename to be saved ')

    argparser.add_argument(
        '--net_id',
        default= 3,
        help='nn id')

    argparser.add_argument(
        '--arm_core',
        default= 'M55',
        help= 'M4 or M55')

    argparser.add_argument(
        '--folder_c',
        default= ".",
        type=str,
        help='C folder')

    argparser.add_argument(
        '--net_name',
        default= 'se',
        help='starting epoch')

    main(argparser.parse_args())
