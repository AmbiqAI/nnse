"""
Test trained NN model using wavefile as input
"""
import os
import pickle
import numpy as np
import c_code_table_converter
from .nn_activation import linear, sigmoid, relu6 # pylint: disable=no-name-in-module
from .nn_module import NeuralNetClass, lstm_states
from .feature_module import FeatureClass
from .converter_fix_point import fakefix
from .load_nn_arch import load_nn_arch, setup_nn_folder
from .log_module import log10_fakeFix

class NNInferClass:
    """
    Class to handle NNInferClass model
    """
    def __init__(
            self,
            nn_arch,
            epoch_loaded,
            params_audio,
            quantized=False,
            show_histogram=False,
            np_inference=False,
            feat_type='mel'):

        self.params_audio = params_audio
        self.show_histogram = show_histogram
        self.np_inference = np_inference
        self.feat_type = feat_type
        out = load_nn_arch(nn_arch)

        neurons, _, layer_types, activations, num_context, self.num_dnsampl = out

        folder_nn = setup_nn_folder(nn_arch)

        nn_infer = NeuralNetClass(
            neurons     = neurons,
            layer_types = layer_types,
            activations = activations,
            batchsize   = 1,
            nDownSample = self.num_dnsampl,
            kernel_size = num_context)

        nn_infer.load_weights(
                f'{folder_nn}/checkpoints/model_checkpoint_ep{epoch_loaded}' )

        if quantized:
            nn_infer.quantized_weight()

        self.nn_infer = nn_infer

        self.states = lstm_states(
            self.nn_infer,
            batchsize=1,
            zero_state = True)

        self.feature_inst = FeatureClass(
                            win_size        = self.params_audio['win_size'],
                            hop             = self.params_audio['hop'],
                            len_fft         = self.params_audio['len_fft'],
                            sample_rate     = self.params_audio['sample_rate'],
                            nfilters_mel    = self.params_audio['nfilters_mel'])

        with open(os.path.join(folder_nn,'stats.pkl'), "rb") as file:
            self.stats = pickle.load(file)
        if self.feat_type=='mel':
            shape = (num_context, self.params_audio['nfilters_mel'])
        elif self.feat_type=='pspec':
            shape = (num_context, 257)

        self.feats_init = np.log10(np.full(shape, 2**-15, dtype=np.float32),
                                    dtype=np.float32)
        self.feats_init = (self.feats_init - self.stats['nMean_feat']) * self.stats['nInvStd']
        self.feats_init = fakefix(self.feats_init, 16, 8)
        self.feats = self.feats_init
        nn_np = c_code_table_converter.tf2np(nn_infer)
        if self.show_histogram:
            c_code_table_converter.draw_nn_hist(nn_np)

        self.nn_np = nn_np
        self.count_run = 0

        if self.np_inference:
            self._act = {
                'tanh'      : np.tanh,
                'sigmoid'   : sigmoid,
                'linear'    : linear,
                'relu6'     : relu6}
            self.states_np = self.lstm_states_np()

    def lstm_states_np(self):
        """
        Initialize lstm state in numpy
        """
        states_np = {'hstate':[], 'cstate':[]}
        for i, neuron in enumerate(self.nn_infer.neurons[1:]):
            if self.nn_infer.layer_types[i]=='lstm':
                states_np['hstate'] += [np.zeros(neuron, dtype=np.float32)]
                states_np['cstate'] += [np.zeros(neuron, dtype=np.float32)]
            else:
                states_np['hstate'] += [None]
                states_np['cstate'] += [None]
        return states_np

    def reset(self):
        """
        Reset states inside instance
        """
        self.feature_inst.reset()
        self.feats          = self.feats_init
        self.states         = lstm_states(self.nn_infer, batchsize=1, zero_state = True)
        self.count_run      = 0
        if self.np_inference:
            self.states_np = self.lstm_states_np()

    def inference_np(self, d_in, hstates, cstates):
        """
        NN inference using numpy
        """
        out = d_in.copy()
        for idx in range(self.nn_infer.num_layers):
            layer_type  = self.nn_infer.layer_types[idx]
            activation  = self.nn_infer.activaitons[idx]
            kernel      = self.nn_np[idx]['kernel']
            bias        = self.nn_np[idx]['bias']
            h_state     = hstates[idx]
            c_state     = cstates[idx]

            if layer_type =='fc':
                pre = np.matmul(out, kernel) + bias
                out = self._act[activation](pre)

            elif layer_type =='conv1d':
                out = out.flatten()
                pre = np.matmul(out, kernel) + bias
                out = self._act[activation](pre)

            elif layer_type=='lstm':
                out = np.concatenate((out, h_state))
                pre = np.matmul(out, kernel) + bias
                i_state, j_state, f_state, o_state = np.split(pre, 4, axis=-1)  # pylint: disable=unbalanced-tuple-unpacking
                i_state = self._act['sigmoid'](i_state)
                j_state = self._act['tanh'](j_state)
                f_state = self._act['sigmoid'](f_state)
                o_state = self._act['sigmoid'](o_state)
                c_state = f_state * c_state + i_state * j_state
                out = o_state * self._act['tanh'](c_state)
                hstates[idx] = out
                cstates[idx] = c_state

        return out, hstates, cstates

    def feature_proc(self, data):
        """
        feature extraction
        """
        spec, _, feat, pspec = self.feature_inst.frame_proc(data)
        if self.feat_type=='pspec':
            feat = log10_fakeFix(pspec)
        feat = fakefix((feat - self.stats['nMean_feat']) * self.stats['nInvStd'], 16, 8)
        self.feats = self.feats[1:]
        self.feats = np.concatenate((self.feats, np.expand_dims(feat, axis=0)), axis=0)
        return feat, spec

    def frame_proc_tf(self, data,
                    return_all=False):
        """
        NN frame process using tensorflow
        """
        feat, spec = self.feature_proc(data)
        feats_expand = np.expand_dims(self.feats, axis=0)

        if self.count_run == 0:

            est, self.states = self.nn_infer(feats_expand, 1.0, self.states, training=False)
            est = est[0,0].numpy()
            self.post_nn_infer(est)
        if return_all:
            return feat, spec, est
        else:        
            return feat, spec

    def frame_proc_np(self, data, return_all=False):
        """
        NN frame process using numpy
        """
        feat, spec = self.feature_proc(data)
        if self.count_run == 0:
            out  = self.inference_np(
                        self.feats,
                        self.states_np['hstate'],
                        self.states_np['cstate'])
            est, self.states_np['hstate'], self.states_np['cstate'] = out
            self.post_nn_infer(est)
        if return_all:
            return feat, spec, est
        else:
            return feat, spec

    def post_nn_infer(self, nn_output):
        """
        post nn inference
        """
        pass
