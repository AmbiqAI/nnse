# Neural Network Speech Enhancement (NNSE)  

Neural Network Speech Enhancement (NNSE) is a speech enhancement (SE) model based on recurrent neural networks (RNN).  

## Getting Started
If you're new to NNSE, begin by cloning the repo and checking out the Python tools for model conversion and training:
```sh
git clone https://github.com/AmbiqAI/nnse.git
cd nnse
git checkout m55_mve
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
To work on Apollo5, you need
- Arm GNU Toolchain 14.2
- Segger J-Link v7.56+
- Python 3.10+

1. Navigate to the Python directory
    ```sh
    cd nnse/python
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
### ✅ **Deployment**
1.  Follow the steps below  
    ```sh
    cd nnse/evb
    make # build
    make deploy # deployment
    make view # will provide SWO output as the device is running.
    ``` 
    Prepare two USB cables. Ensure your board is connected via both the `JLINK USB port` and the `audio USB port`. Then turn on the power on EVB.
1. Plug a mic into the 3.5mm port.

1. In a separate terminal, activate the Python environment and launch the 10-second recording tool:
    ```sh
    cd nnse/python/tools  # Navigate to the "nnse/python" directory
    ```
1. Run the appropriate command based on your operating system:
    ```sh
    # macOS
    python -m record_evb --tty /dev/tty.usbmodem1234561

    # Ubuntu
    python -m record_evb --tty /dev/serial/by-id/usb-TinyUSB_TinyUSB_Device_123457-if00

    # Windows
    python -m record_evb --tty COM4
    ```
1. Start SE Recording
    - Press Button 1 on the EVB to begin a speech enhancement (SE) recording session.
    - Speak clearly into the microphone while the recording is active.
    - Press Button 1 again to stop the recording.


1. Check the two recording files under `nnse/python/tools/audio_result/`. 
    - `rpc_audio_raw.wav`: the raw PCM data from your mic.
    - `rpc_audio_en.wav`: the enhanced speech.
