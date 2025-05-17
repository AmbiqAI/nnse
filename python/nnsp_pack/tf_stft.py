""" stft and istft functions for tensorflow
    Those function support batch processing
"""
import tensorflow as tf
import numpy as np
def gen_stft_win(
        win_size: int     = 240,
        hop : int         = 80,
        len_fft: int      = 256):
    """
    STFT window generator
    """
    win_size = np.float32(win_size)
    hop_div_fr = np.float32(hop / win_size)
    indices = np.arange(win_size, dtype=np.float32)
    # import pdb; pdb.set_trace()
    win_square = hop_div_fr * (1.0 - np.cos(2.0 * np.pi / win_size * indices ))
    win = np.sqrt(win_square)

    return win

def window_fn(
        frame_length: int,
        dtype=tf.float32):
    """ Create a window function"""
    win = gen_stft_win(
            win_size        = frame_length,
            hop             = frame_step,
            len_fft         = fft_length)

    return tf.constant(win, dtype=dtype)

def tf_stft(
        signals: tf.float32,
        frame_length: int,
        frame_step: int,
        fft_length: int,
        window_fn):
    """ Calculate the STFT of a batch of signals"""
    shape = tf.shape(signals)
    zeros = tf.zeros([shape[0], frame_step*2])
    signals = tf.concat([zeros, signals], axis=-1)
    signals_stft = tf.signal.stft(
        signals,
        frame_length=frame_length,  # Length of each frame in samples
        frame_step=frame_step,   # Number of samples to shift between frames
        window_fn=window_fn,
        fft_length=fft_length     # Length of the FFT
    )
    return signals_stft

def tf_istft(
        signals_stft: tf.float32,
        frame_length: int,
        frame_step: int,
        fft_length: int,
        window_fn):
    """ Calculate the inverse STFT of a batch of signals"""

    signals_recons = tf.signal.inverse_stft(
        signals_stft,
        frame_length = frame_length,
        frame_step = frame_step,
        fft_length=fft_length,
        window_fn=window_fn,
        name=None
    )
    return signals_recons[:,frame_step*2:-frame_step*2]

if __name__ == '__main__':
    frame_length = 320
    frame_step = 160
    fft_length = 512
    signals = tf.random.normal([2, 160000])  # Batch of 2 signals, each with 16000 samples
    signals_stft = tf_stft(
        signals,
        frame_length,
        frame_step,
        fft_length,
        window_fn)

    print(signals_stft.shape)
    print(signals_stft)

    signals_recons = tf_istft(
        signals_stft,
        frame_length,
        frame_step,
        fft_length,
        window_fn)
    print(signals_recons.shape)
    print(signals_recons)
    print('reconstruction error:')
    error = signals[:,:-320] - signals_recons
    print(tf.reduce_max(tf.abs(error)))

    import matplotlib.pyplot as plt
    plt.subplot(3,1,1)
    plt.plot(signals[0,:-320])
    plt.subplot(3,1,2)
    plt.plot(signals_recons[0])
    plt.subplot(3,1,3)
    plt.plot(signals[0,:-320] - signals_recons[0])
    plt.ylim([-1, 1])
    plt.show()