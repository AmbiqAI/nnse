# NNSE -- Neural Noise Suppression Engine

This project provides a lightweight **Neural Noise Suppression Engine
(NNSE)** with a simple C interface for real-time audio processing. It is
designed to work with 16 kHz audio streams in 10 ms frames (160 samples
per frame).

------------------------------------------------------------------------

## 📦 Project Structure

    nnse/
    ├── *.c, *.h             # wrapper sources
    ├── ns-nnsp/             # Library source code
    │   ├── src/             # Core implementation
    │   └── includes-api/    # Public API headers
    ├── README.md            # This file

------------------------------------------------------------------------

## ⚙️ Requirements

-   **Compiler**: GCC 14.2 (default)
-   **Optimization**: `-O3` recommended
-   **Language**: C (C99 or later)
-   **Target Architecture**:
    -   ARM **Cortex-M55 with MVE (Helium)** support for optimized
-   **Dependencies**: CMSIS-DSP-1.16.2 library is required for efficient signal processing operations

------------------------------------------------------------------------

## 🚀 Basic Usage

``` c
#include "AudioPipe_wrapper.h"

int16_t pcm_input[160];  // 10ms @ 16kHz
int16_t pcm_output[160];

int main() {
    AudioPipe_wrapper_init();
    AudioPipe_wrapper_reset();

    // Streaming loop
    while (1) {
        AudioPipe_wrapper_frameProc(pcm_input, pcm_output);
    }
}
```
## 🏗️ Building the NS-NNSP Library

The NS-NNSP library provides the core implementation for the NNSE.
To build it, simply navigate to the ns-nnsp directory and run:
```sh
cd ns-nnsp
make
```
This will generate the compiled library (.a) inside the build output folder.

You can then link this library against your application sources when building your project.

Alternatively, you can directly compile the sources when building your application



### API Overview

-   **`AudioPipe_wrapper_init()`**\
    Initializes the noise suppression engine.
-   **`AudioPipe_wrapper_reset()`**\
    Resets the engine state (useful for re-initialization).
-   **`AudioPipe_wrapper_frameProc(int16_t* in, int16_t* out)`**\
    Processes a 10 ms audio frame. Input/output are PCM16 arrays of
    length 160.

------------------------------------------------------------------------

