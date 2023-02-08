"""
Activation functions for Neural networks
"""
import numpy as np
import matplotlib.pyplot as plt
def linear(data):
    """
    linear activation function
    """
    return data

def sigmoid(data):
    """
    sigmoid activation function
    """
    return 1.0 / (1.0 + np.exp(-data))

def relu6(data):
    """
    relu6 activatioin function
    """
    return np.minimum(np.maximum(data, 0), 6)

def softmax(data):
    """softmax function"""
    data_o = np.exp(data)/sum(np.exp(data))
    return data_o

def main():
    """
    Test some activation functions
    """
    indata = np.linspace(-10, 10, 1000)
    plt.plot(indata, sigmoid(indata))
    plt.plot(indata, relu6(indata))
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
