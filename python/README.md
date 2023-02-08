# NN SE (NNSE) model
This document explains how we train speech enhancement (SE) by using recurrent neural network.
## Prerequisite
- Python 3.7+

Note that all python scripts described here are all under the folder `nnse/python`
1. (optional but recommended) Create a python [virtualenv](https://docs.python.org/3/library/venv.html) and install python dependencies into it:
- Linux
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# call other python tasks defined below with this active
# then when finished with this virtualenv type:
deactivate
```
- Windows: in command window, type
```cmd
python -m venv .venv
.venv/Scripts/activate.bat
pip install -r requirements.txt
# call other python tasks defined below with this active
# then when finished with this virtualenv type:
deactivate
```
## Quick start
If you want to test the already trained model, please go to `step 3` directly.
## Training procedure
1. Feature extraction and save your features as tfrecord (see [here](https://www.tensorflow.org/guide/data) and [here](https://www.tensorflow.org/guide/data_performance)). Type
```cmd
  $ python data_se.py                            
```
2. Train your model. Type
```cmd
  $ python train_se.py --epoch_loaded='random' --nn_arch='nn_arch/def_se_nn_arch72_mel.txt' 
```
  * The argument `--epoch_loaded` represents which epoch of the weight table to be loaded
    - `--epoch_loaded='random'`means you start to train NN from a   randomly initiialized set of weights
    - `--epoch_loaded='latest'`means you start to train NN from the lateset set of weights of that epoch to be saved
    - `--epoch_loaded=10` (or any non-negative integer) means we will attempt to load a model from the previously saved epoch=10 if it exists.
  * The argument `--nn_arch='nn_arch/def_se_nn_arch72_mel.txt'` will load the definition of NN architecture in `nn_arch/def_se_nn_arch72_mel.txt` (see [here](nn_arch/def_s2i_nn_arch.txt)). Also, the trained model is saved in the folder `models_trained/def_se_nn_arch72_mel`. Note that the foldername `def_se_nn_arch72_mel` is the same as definition of nn architecture, `def_se_nn_arch72_mel.txt`, except of removing the prefix `def_` and suffix `.txt`.
    - `NN architecture`: our nn architecture only supports sequential model (see the example [here](nn_arch/def_s2i_nn_arch.txt)). 
      - The layer type supports `fc`, `lstm`, `conv1d`
      - Activation type supports `relu6`, `tanh`, `sigmoid`, `linear`

3.  Test from recorded wave file. Type
```cmd
  $ python test_se.py --epoch_loaded=85 --nn_arch='nn_arch/def_se_nn_arch72_mel.txt' --recording=1
```
  * Here we provide an already trained model. Its nn architecture is defined in `nn_arch/def_se_nn_arch72_mel.txt`. You can change to your own model later.
  * The argument `--nn_arch='nn_arch/def_se_nn_arch72_mel.txt'` will load the definition of NN architecture in `nn_arch/def_se_nn_arch72_mel.txt`. 
  * The argument `--epoch_loaded=85` means it will load the model saved in epoch = 85.
  * The argument `--recording=1` means it will, first, record your speech for 10 seconds and save it in `test_wavs/speech.wav`. Second, use `test_wavs/speech.wav` as input to run the inference and check its result.
  * Alternatively, you can run the already saved wave file via setting `--recording=0`
```cmd
  $ python test_se.py --epoch_loaded=85 --nn_arch='nn_arch/def_se_nn_arch72_mel.txt' --recording=0 --test_wavefile='test_wavs/speech.wav'
```
  * This will directly use the already saved wave file `--test_wavefile='test_wavs/speech.wav'` without recording.

# Convert TF-model to C table
To run the model on the embedded system, Apollo4 in our cae, we need a tool to support
1. A neural network architecture, equivalent to Tensorflow, to perform on the desired microcontroller,
2. A converter to convert the set of weight tables trained by Tensorflow to save on the memory of the microcontroller. One of the common format to save weight table would be `8-bit` integer, and this is what we adopted as well. 

For example, [TFLM](https://www.tensorflow.org/lite/microcontrollers) (tensorflow lite for microcontroller) supports two functionalities.

In Tensorflow NN training, we usually use `32-bit floating` point format (or even higher precision such as biases)  to save the weight table or activations. However, it is not very friendly for microcontroller due to the limitation of memory size and computational power. To overcome this shortage, one of the most common format is to quantize them to the lower precision scheme, such as `8-bit integer`. However, this might degrade the performance due to the simple compression scheme. The [quantization aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training) is advised to mitigate the degradation. 

Furthermore, as mentioned in [TFLM Post-training integer quantization with int16 activations](https://www.tensorflow.org/lite/performance/post_training_integer_quant_16x8), by making activations to `16-bit` integer values, while keeping weight table as 8-bit integer, this can improve accuracy of the quantized model significantly, compared to the `8-bit` integer activation case. Tensorflow refers to this mode as the `16x8 quantization mode`. However, as we tried the `16x8 quantization mode` on Apollo4, it does not work. The reason might be that `16x8 quantization mode` is still in the experimental stage.  

To resolve this deficiency, 
1. we provide the C library, `ns-nnsp.a` under the folder `../evb/libs/`, to support the neural network with activations in `16-bit` values while keep the weight table in 8-bit integer values.
2. we provide a converter, [c_code_table_converter.py](./c_code_table_converter.py), to quantize the weight table in 8-bit integer format and convert it to C format so that `ns-nnsp.a` can call for. 
## C table conversion
 To convert the trained model by Tensorflow to C table, type:
```cmd
  $ python c_code_table_converter.py --epoch_loaded=85 --nn_arch='nn_arch/def_se_nn_arch72_mel.txt' --net_id=3 --net_name='se'
```
  * Here we provide an already trained model. Its nn architecture is defined in `nn_arch/def_se_nn_arch72_mel.txt`. You can change to your own model later.
  * The argument `--s2i_nn_arch='nn_arch/def_se_nn_arch72_mel.txt'` will load the definition of NN architecture in `nn_arch/def_se_nn_arch72_mel.txt`. 
  * The argument `--epoch_loaded=85` means it will load the model saved in epoch = 800.
  * The argument `--net_name='se'` provides the this neural net a specific name. This is very important if you use sevearal NNs.Ensure that you only assign each NN one and only one `net_name`.
  * The argument `--net_id=3` provides the NN an identification. Ensure that you only assign each NN one and only one `net_id`.
  
  After execute `c_code_table_converter.py`, you can see that it generates two files as below
  ```cmd
../evb/src/def_nn3_se.h
../evb/src/def_nn3_se.c
  ```
Note that the header (\*.h) and the source file (\*.c) follow the rules below 
```c
def_nn{net_id}_{net_name}.h and
def_nn{net_id}_{net_name}.c
```
The `def_nn3_se.c` saves the set of weight tables and its NN architecture in this neural net, called `nn3_se` here.

# Deploy to Apollo4
See [README.md](../README.md) here.