# Neural Network Speech Enhancement (NNSE) model
This document explains how we train speech enhancement (SE) by using recurrent neural network.
# Loss functions
It is worth to compare the performance based on the loss functions listed in Table 1 (see [here](https://www.microsoft.com/en-us/research/uploads/prod/2021/08/23.pdf)).
## Prerequisite
Note that all python scripts described here are all under the folder `nnse/python`
- Python 3.7+
- (optional but recommended) Create a python [virtualenv](https://docs.python.org/3/library/venv.html) and install python dependencies into it:
  - Linux
    ```sh
    $ cd nnse/python
    $ python -m venv .venv
    $ source .venv/bin/activate
    $ ./setup.sh
    # call other python tasks defined below with this active
    # then when finished with this virtualenv type:
    deactivate
    ```
  - Windows: in command window, type
    ```cmd
    $ cd nnse/python
    $ python -m venv .venv
    $ .venv/Scripts/activate.bat
    $ pip install -r requirements.txt
    # call other python tasks defined below with this active
    # then when finished with this virtualenv type:
    deactivate
    ```
## Dataset
Before working on training SE model, we need to download the required datasets. Please read on their license agreements carefully in [here](../docs/README.md).
## Quick start
We provided two already trained models. The user can directly try on it.
- Simple RNN: `Small size model:` `~100k` parameters
  ```bash
    $ python train_se.py \
      --mode test \
      --test_wavefile test_wavs/keyboard_steak.wav \
      --epoch_loaded 50 \
      --config_file nn_arch/config_se_nn_arch72_mel.yaml 
  ```
  `Input argruments`:
    * `--mode test`: test on a already trained model
    * `--test_wavefile test_wavs/keyboard_steak.wav`: `test_wavs/keyboard_steak.wav` will be tested (16kHz sampling rate)
    * `--config_file nn_arch/config_se_nn_arch72_mel`: it will load the definition of NN architecture in [config_se_nn_arch72_mel.yaml](nn_arch/config_se_nn_arch72_mel.yaml). 
    * `--epoch_loaded 50`: it will load the model saved in epoch = 50.
  
  `Outputs:`
  * The enhanced speech is located at `test_results/config_se_nn_arch72_mel/keyboard_steak/enhanced_speech.wav`.
  * It also shows the speech quality
  ```sh
  ---Noisy---
  STOI: tensor([0.6946], grad_fn=<SqueezeBackward1>)
  PESQ: tensor([1.1136], grad_fn=<SqueezeBackward1>)
  SI-SDR: tensor([4.3818], grad_fn=<SqueezeBackward1>)

  ---Enhance---
  STOI: tensor([0.7852], grad_fn=<SqueezeBackward1>)
  PESQ: tensor([1.1945], grad_fn=<SqueezeBackward1>)
  SI-SDR: tensor([8.3645], grad_fn=<SqueezeBackward1>)
  ```
- Unet: 
    ```cmd
    $ python train_se.py \
      --mode test \
      --test_wavefile test_wavs/keyboard_steak.wav \
      --epoch_loaded 117 \
      --config_file nn_arch/config_unet_relu_noncausal_sep_specmel_th50.yaml
    ```
    * 
    ```sh
    ---Noisy---
    STOI: tensor([0.6946], grad_fn=<SqueezeBackward1>)
    PESQ: tensor([1.1136], grad_fn=<SqueezeBackward1>)
    SI-SDR: tensor([4.3818], grad_fn=<SqueezeBackward1>)

    ---Enhance---
    STOI: tensor([0.8025], grad_fn=<SqueezeBackward1>)
    PESQ: tensor([1.2402], grad_fn=<SqueezeBackward1>)
    SI-SDR: tensor([9.3557], grad_fn=<SqueezeBackward1>)
    ```
 

## Training procedure
1. Feature extraction and save your features as tfrecord (see [here](https://www.tensorflow.org/guide/data) and [here](https://www.tensorflow.org/guide/data_performance)). Type
    ```cmd
      $ python data_se.py --download 1 --dataset_noise 30000                     
    ```
    * `--download`:
      * `--download 1`: it will automatically download all of the training data and then start to work on feature extraction.
      * `--download 0`: it will assume dataset had been downloaded and start to work on feature extraction.
    * `--datasize_noise`: the size of training dataset per noise, e.g.,
      * `--datasize_noise 30000`: it randomly chooses 30000 speech samples on the training dataset (total size of training dataset is `93188`)
      * `--datasize_noise -1`: it uses the total size of training dataset 
      (`93188` speech samples)
            
2. Train your model. Type
    ```cmd
      $ python train_se.py --epoch_loaded random --config_file nn_arch/config_se_nn_arch72_mel.yaml
    ```
    * The argument `--epoch_loaded` represents which epoch of the weight table to be loaded
      - `--epoch_loaded 'random'`means you start to train NN from a   randomly initiialized set of weights
      - `--epoch_loaded 'latest'`means you start to train NN from the lateset set of weights of that epoch to be saved
      - `--epoch_loaded 10` (or any non-negative integer) means we will attempt to load a model from the previously saved epoch=10 if it exists.
    * The argument `--config_file nn_arch/config_se_nn_arch72_mel.yaml` will load the definition of NN architecture and type of feature extraction in `nn_arch/config_se_nn_arch72_mel.yaml` (see [here](nn_arch/config_se_nn_arch72_mel.yaml)). Also, the trained model is saved in the folder `models_trained/se_nn_arch72_mel`. Note that the foldername `se_nn_arch72_mel` is the same as definition of nn architecture, `config_se_nn_arch72_mel.yaml`, except of removing the prefix `config_` and suffix `.yaml`.
3. Monitor your training using Tensorboard. Open a new shell and type
    ```bash
    $ tensorboard --logdir tensorboard
    ```
    Open your web browser and enter the address (http://localhost:6006/)

# Deploy Your Model to Ambiq Microcontrollers Apollo4 and Apollo5
  See [README.md](../README.md).