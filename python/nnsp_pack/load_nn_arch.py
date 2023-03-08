"""
This module works on parsing nn architecture and setup its folder
"""
import os
import re

def load_nn_arch(filename):
    """
    Load NN architecture, including layer_type, activation, neuron.
    All of nn_architecture name sould start from def_ and end with .txt
    """
    def nn_search(targets, lines, dtype):
        output=[]
        for line in lines:
            line = line.strip()
            line = re.sub(r'#.*', '', line)
            if line != '':
                sps = line.split()
                if sps[0] in targets:
                    output +=[dtype(sps[1])]
        return output

    with open(filename, 'r') as file: # pylint: disable=unspecified-encoding
        lines = file.readlines()

    size_neurons = nn_search(('layer_neurons', 'num_filters'), lines, dtype=int)
    tmp = nn_search(('input_neurons'), lines, dtype=int)
    size_neurons = tmp + size_neurons

    dropprobs = nn_search(('dropprob'), lines, dtype=float)
    layer_types = nn_search(('layer_type'), lines, dtype=str)
    activations = nn_search(('activation'), lines, dtype=str)
    try:
        num_context = nn_search(('kernel_size'), lines, dtype=int)[0]
    except:
        num_context = 1
    try:
        num_dnsampl = nn_search(('strides'), lines, dtype=int)[0]
    except:
        num_dnsampl = 1

    print(f"\nLoad nn_arch={filename}:")
    print("neurons:", size_neurons)
    print("Dropout probs:", dropprobs)
    print("Layer types:", layer_types)
    print("Activations:", activations, '\n')

    return size_neurons, dropprobs, layer_types, activations, num_context, num_dnsampl

def setup_nn_folder(nn_arch_name):
    """
    Setup the NN_folder based on nn_arch_name.
    NN_folder name will remove the prefix def_ and suffix .txt of nn_arch_name.

    """
    folder = re.sub(r'def_', '', nn_arch_name)
    folder = re.sub(r'\.txt$', '', folder)
    folder_nn = f'models_trained/{os.path.basename(folder)}'
    os.makedirs(folder_nn, exist_ok=True)
    os.makedirs(f'{folder_nn}/checkpoints', exist_ok=True)
    return folder_nn

def main():
    """
    Testing
    """
    fname='nn_arch/def_s2i_nn_arch.txt'
    load_nn_arch(fname)

if __name__ == '__main__':
    main()
    