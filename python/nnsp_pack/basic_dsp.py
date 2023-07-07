import numpy as np
import scipy
from nnsp_pack.converter_fix_point import fakefix
def dc_remove(inputs: np.float32)->np.float32:
    """remove audio dc term

    Args:
        x (float32): input

    Returns:
        float32: output
    """
    filter = scipy.signal.lfilter
    b_hpf = np.array([1.0, -1.0])
    a_hpf = np.array([1.0, -0.965267479])
    a_hpf = fakefix(a_hpf, 16, 15)
    outputs = filter(b_hpf, a_hpf, inputs)
    return outputs

def dc_remove1(inputs: np.float32)->np.float32:
    """remove audio dc term

    Args:
        x (float32): input

    Returns:
        float32: output
    """
    b = -1
    a = fakefix(-0.965267479, 16,15)
    g = 0
    output = inputs.copy() * 0
    for i in range(len(inputs)):
        tmp =   fakefix(inputs[i] - a * g, 32, 15)
        output[i] = fakefix(g + b * tmp, 16, 15)
        g = tmp
    return output

def pre_emphasis(inputs: np.float32)->np.float32:
    """pre-emphasis

    Args:
        x (float32): input

    Returns:
        float32: output
    """
    filter = scipy.signal.lfilter
    b_hpf = np.array([1.0, -0.97])
    a_hpf = np.array([1.0])
    outputs = filter(b_hpf, a_hpf, inputs)
    return outputs

def de_emphasis(inputs: np.float32)->np.float32:
    """de-emphasis

    Args:
        x (float32): input

    Returns:
        float32: output
    """
    filter = scipy.signal.lfilter
    b_hpf = np.array([1.0])
    a_hpf = np.array([1.0,-0.97])
    outputs = filter(b_hpf, a_hpf, inputs)
    return outputs

if __name__=="__main__":
    print("basic_dsp.py")
    x = np.array([1.0, -5, 8, 4, -2])
    y = pre_emphasis(x)
    z = de_emphasis(y)
    e = np.abs(x-z).max()
    print(f"x=\n{x}")
    print(f"pre_emp=\n{y}")
    print(f"de_emp=\n{z}")
    print(f"error={e}")
    