""" stft and istft functions for tensorflow
    Those function support batch processing
"""
import tensorflow as tf
import numpy as np
import math as m
pi = tf.constant(m.pi)
def gen_stft_win(
        win_size: int     = 240,
        hop : int         = 80):
    """
    STFT window generator
    """
    win_size = tf.cast(win_size, tf.float32)
    hop_div_fr = tf.cast(hop,tf.float32) / win_size
    indices = tf.range(win_size, dtype=tf.float32)
    # import pdb; pdb.set_trace()
    win_square = hop_div_fr * (1.0 - tf.math.cos(2.0 * pi / win_size * indices ))
    win = tf.sqrt(win_square)

    return win

def window_fn(
        frame_length: int,
        frame_step: int,
        dtype=tf.float32):
    """ Create a window function"""
    win = gen_stft_win(
            win_size        = frame_length,
            hop             = frame_step)

    return win
@tf.function
def tf_stft(
        signals: tf.float32,
        frame_length: int,
        frame_step: int,
        fft_length: int):
    """ Calculate the STFT of a batch of signals"""
    def my_win(frame_length, dtype=tf.float32):
        return window_fn(frame_length, frame_step,dtype=dtype)
    shape = tf.shape(signals)
    zeros = tf.zeros([shape[0], frame_length-frame_step])
    signals = tf.concat([zeros, signals], axis=-1)
    signals_stft = tf.signal.stft(
        signals,
        frame_length=frame_length,  # Length of each frame in samples
        frame_step=frame_step,   # Number of samples to shift between frames
        window_fn=my_win,
        fft_length=fft_length     # Length of the FFT
    )
    return signals_stft

@tf.function
def tf_istft(
        signals_stft: tf.float32,
        frame_length: int,
        frame_step: int,
        fft_length: int):
    """ Calculate the inverse STFT of a batch of signals"""
    def my_win(frame_length, dtype=tf.float32):
        return window_fn(frame_length, frame_step,dtype=dtype)
    
    signals_recons = tf.signal.inverse_stft(
        signals_stft,
        frame_length = frame_length,
        frame_step = frame_step,
        fft_length=fft_length,
        window_fn=my_win,
        name=None
    )
    overlap=frame_length-frame_step
    return signals_recons[:,overlap:-overlap]

if __name__ == '__main__':
    frame_length = 480
    frame_step = 160
    fft_length = 512
    signals = tf.random.normal([2, 160000])  # Batch of 2 signals, each with 16000 samples
    signals_stft = tf_stft(
        signals,
        frame_length,
        frame_step,
        fft_length)

    print(signals_stft.shape)
    print(signals_stft)

    signals_recons = tf_istft(
        signals_stft,
        frame_length,
        frame_step,
        fft_length)
    print(signals_recons.shape)
    print(signals_recons)
    print('reconstruction error:')

    overlap=frame_length-frame_step
    error = signals[:,:-overlap] - signals_recons
    print(tf.reduce_max(tf.abs(error)))

    import matplotlib.pyplot as plt
    plt.subplot(3,1,1)
    plt.plot(signals[0,:-overlap])
    plt.subplot(3,1,2)
    plt.plot(signals_recons[0])
    plt.subplot(3,1,3)
    plt.plot(signals[0,:-overlap] - signals_recons[0])
    plt.ylim([-1, 1])
    plt.show()
