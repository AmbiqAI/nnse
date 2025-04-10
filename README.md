# Neural Network Speech Enhancement (NNSE)  

Neural Network Speech Enhancement (NNSE) is a speech enhancement (SE) model based on recurrent neural networks (RNN).  

**Update (04/02/2025):** 
1. Added support for the U-Net architecture.
2. Added TensorFlow Lite for Microcontrollers (TFLM) support (16x8 bit).

## Getting Started
If you're new to NNSE, begin by cloning the repo and checking out the Python tools for model conversion and training:
```sh
git clone https://github.com/AmbiqAI/nnse.git
```
This will download the nnse directory.
## Directory contents
```py
nnse/ # root 
    evb/ # for evb deployment
        build/      # bin files
        includes/   # required inlcudes
        libs/       # required libs
        make/       # make.mk
        pack/
        src/        # c source codes
        Makefile
        autogen.mk
    python/   # for NN training
    README.md # this readme
```
## Prerequisite
### `Software`
To work on Apollo4, you need
- Arm GNU Toolchain 14.2
- Segger J-Link v7.56+
# Speech Enhancement
The speech enhancement models are based on 16 kHz sampling rate. The model sizes are about 100kB (simple RNN) and 170kB (U-Net).

### `Dataset`
The SE model is trained based on several audio dataset, including human speech and noises. Before you use this repo, please read on their license agreements carefully in [here](./docs/README.md).



## Compiling and Running a Pre-Trained Model  

There are two main options for deploying TensorFlow models to Ambiq microcontrollers:

1. TensorFlow Lite for Microcontrollers (TFLM)
    - Broad support for standard neural network layers.

    - The ns-tflm library extends TFLM to handle certain 16x8-bit layers, such as separable convolution and LSTM.

1. NeuralSPOT NNSP (NS-NNSP)

    - A streamlined, in-house framework designed for high efficiency.

    - Supports a narrower range of layers: conv1d, LSTM, and fully connected (FC).

### Using TFLM
To convert a model to TensorFlow Lite for Microcontrollers (TFLM) for Apollo 510 or Apollo 4, you can use neuralSPOT, an AI development toolkit provided by AmbiqAI.

We recommend downloading **neuralSPOT** in the same directory as **nnse**, maintaining the following directory structure:  

```sh
📂 target-directory/
   ├── 📂 nnse/
   ├── 📂 neuralSPOT/
```  

This setup ensures seamless integration and compatibility.

To clone the neuralSPOT repository, use the following commands:
```sh
cd <target-directory>
git clone https://github.com/AmbiqAI/neuralSPOT.git
```

This setup ensures compatibility and ease of integration with nnse.

To generate a TensorFlow Lite (TFLite) model from a pre-trained model, follow these steps:
1. Navigate to the Python directory
    ```sh
    cd nnse/python
    python -m venv .venv
    source .venv/bin/activate
    ./setup.sh
    ```
1. Define the TFLite filename to be saved
    ```sh
    tflite_filename=nnse_rnn_int16
    ```
1. Run the TFLite conversion script
    
    **🔹 For Simple RNN Model**
    ```sh
    python c_code_table_converter.py --is_tflite 1 \
      --config_file nn_arch/config_se_nn_arch72_mel.yaml \
      --epoch_loaded 50 \
      --tflite_filename ./$tflite_filename.tflite
    ```
    or \
    **🔹 For U-Net Model**
    ```sh
    python c_code_table_converter.py --is_tflite 1 \
      --config_file nn_arch/config_unet_relu_noncausal_sep_specmel_th50.yaml \
      --epoch_loaded 117 \
      --tflite_filename ./$tflite_filename.tflite
    ```
    This script converts the trained model to the TFLite one.
1. Convert the TFLite model to TensorFlow Lite for Microcontrollers (TFLM)
    ```sh
    ./tflm_autodeploy.sh $tflite_filename
    ```
    This script executes the conversion of the TFLite model to a format compatible with TFLM inside `nnse/evb/src/tflm`.
1. Clean and build the project
    ```sh
    # under the directory "nnse/evb"
    cd ../evb
    make clean
    make model_folder=tflm
    ```
### ✅ **Deployment**
1.  Follow the steps below  
    ```sh
    # under the directory "nnse/evb"
    make deploy
    make view # will provide SWO output as the device is running.
    ``` 
    Prepare two USB cables. Ensure your board is connected via both the `JLINK USB port` and the `audio USB port`. Then turn on the power on EVB.
1. Plug a mic into the 3.5mm port.

1. In a separate terminal, activate the Python environment and launch the 10-second recording tool:
    ```sh
    cd nnse/python  # Navigate to the "nnse/python" directory
    source .venv/bin/activate # Activate the virtual environment
    cd tools # Go to "nnse/python/tools" directory
    ```
1. Run the appropriate command based on your operating system:
    ```sh
    # macOS
    python -m record_10s --tty /dev/tty.usbmodem1234561

    # Ubuntu
    python -m record_10s --tty /dev/serial/by-id/usb-TinyUSB_TinyUSB_Device_123457-if00

    # Windows
    python -m record_10s --tty COM4
    ```
1. Press button 1 on EVB.
    - 🎧 This initiates a 10-second SE recording session. 
    - Ensure you're speaking into the microphone during this time.

1. Check the two recording files under `nnse/python/tools/audio_result/`. 
   - `rpc_audio_raw.wav`: the raw PCM data from your mic.
   - `rpc_audio_en.wav`: the enhanced speech.

### Using NNSP
To generate the necessary files for NNSP, follow these steps:
1. Navigate to the Python directory and run the conversion script
    ```sh
    cd nnse/python
    python c_code_table_converter.py --is_tflite 0 \
    --config_file nn_arch/config_se_nn_arch72_mel.yaml \
    --epoch_loaded 50

    ```
1. Copy the generated source files to the target directory
    ```sh
    cp def_nn3_se.{c,h} ../evb/src/nnsp/
    ```
1. Navigate to the `evb` directory
    ```sh
    cd ../evb # go to "nnse/evb"
    ```
1. Clean and build the project
    ```sh
    # under the directory "nnse/evb"
    make clean
    make model_folder=nnsp
    ```
Follow the steps under the ✅ **Deployment** section.

## Re-Training a New Model

Our approach to training the model can be found in [README.md](./python/README.md).

## NS-NNSP Library Overview  

The `ns-nnsp.a` library, currently part of NeuralSPOT, is a C library designed for building pipelines that include feature extraction and neural network processing on the Apollo4/Apollo5 platform. The source code is located in the directory [here](https://github.com/AmbiqAI/neuralSPOT/tree/main/neuralspot/ns-nnsp) and can be modified or rebuilt using [NeuralSPOT, Ambiq's AI Enablement Library](https://github.com/AmbiqAI/neuralSPOT).