#include <stdint.h>
#include "s2iCntrlClass.h"
#include "am_util_stdio.h"
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_audio.h"
#include "ambiq_nnsp_const.h"
#include "arm_intrinsic_test.h"
#include "ns_timer.h"
#include "ns_energy_monitor.h"

#define ENERGY_MEASUREMENT

#define NUM_CHANNELS 1
int volatile g_intButtonPressed = 0;
///Button Peripheral Config Struct
ns_button_config_t button_config_nnsp = {
    .button_0_enable = true,
    .button_1_enable = false,
    .button_0_flag = &g_intButtonPressed,
    .button_1_flag = NULL
};
/// Set by app when it wants to start recording, used by callback
bool volatile static g_audioRecording = false;
/// Set by callback when audio buffer has been copied, cleared by
/// app when the buffer has been consumed.
bool volatile static g_audioReady = false;
/// Audio buffer for application
int16_t static g_in16AudioDataBuffer[LEN_STFT_HOP << 1];
uint32_t static audadcSampleBuffer[LEN_STFT_HOP * 2 + 3];

/**
* 
* @brief Audio Callback (user-defined, executes in IRQ context)
* 
* When the 'g_audioRecording' flag is set, copy the latest sample to a buffer
* and set a 'ready' flag. If recording flag isn't set, discard buffer.
* If 'ready' flag is still set, the last buffer hasn't been consumed yet,
* print a debug message and overwrite.
* 
*/
void
audio_frame_callback(ns_audio_config_t *config, uint16_t bytesCollected) {
    uint32_t *pui32_buffer =
        (uint32_t *) am_hal_audadc_dma_get_buffer(config->audioSystemHandle);

    if (g_audioRecording) {
        if (g_audioReady)
            ns_lp_printf("Warning - audio buffer wasnt consumed in time\n");

        // Raw PCM data is 32b (12b/channel) - here we only care about one
        // channel For ringbuffer mode, this loop may feel extraneous, but it is
        // needed because ringbuffers are treated a blocks, so there is no way
        // to convert 32b->16b
        for (int i = 0; i < config->numSamples; i++) {
            g_in16AudioDataBuffer[i] = (int16_t)( pui32_buffer[i] & 0x0000FFF0);

            if (i == 4) {
                // Workaround for AUDADC sample glitch, here while it is root caused
                g_in16AudioDataBuffer[3] = (g_in16AudioDataBuffer[2] + g_in16AudioDataBuffer[4]) >> 1; 
            }
        }
#ifdef RINGBUFFER_MODE
        ns_ring_buffer_push(&(config->bufferHandle[0]),
                                      g_in16AudioDataBuffer,
                                      (config->numSamples * 2), // in bytes
                                      false);
#endif
        g_audioReady = true;
    }
}

/**
 * @brief NeuralSPOT Audio config struct
 * 
 * Populate this struct before calling ns_audio_config()
 * 
 */
ns_audio_config_t audio_config = {
#ifdef RINGBUFFER_MODE
    .eAudioApiMode = NS_AUDIO_API_RINGBUFFER,
    .callback = audio_frame_callback,
    .audioBuffer = (void *)&pui8AudioBuff,
#else
    .eAudioApiMode = NS_AUDIO_API_CALLBACK,
    .callback = audio_frame_callback,
    .audioBuffer = (void *) &g_in16AudioDataBuffer,
#endif
    .eAudioSource = NS_AUDIO_SOURCE_AUDADC,
    .sampleBuffer = audadcSampleBuffer,
    .numChannels = NUM_CHANNELS,
    .numSamples = LEN_STFT_HOP,
    .sampleRate = SAMPLING_RATE,
    .audioSystemHandle = NULL, // filled in by audio_init()
#ifdef RINGBUFFER_MODE
    .bufferHandle = audioBuf
#else
    .bufferHandle = NULL
#endif
};

const ns_power_config_t ns_lp_audio = {.eAIPowerMode = NS_MAXIMUM_PERF,
                                       .bNeedAudAdc = true,
                                       .bNeedSharedSRAM = false,
                                       .bNeedCrypto = false,
                                       .bNeedBluetooth = false,
                                       .bNeedUSB = false,
                                       .bNeedIOM = false,
                                       .bNeedAlternativeUART = false,
                                       .b128kTCM = false,
                                       .bEnableTempCo = false,
                                       .bNeedITM = false};                                  

int main(void) {
    s2iCntrlClass cntrl_inst;
    g_audioRecording = false;

    ns_core_init();
    ns_power_config(&ns_lp_audio);
    // ns_power_config(&ns_development_default);

    #ifdef ENERGY_MEASUREMENT
        // ns_uart_printf_enable(); // use uart to print, uses less power
        ns_itm_printf_enable(); 
        ns_init_power_monitor_state();
        ns_set_power_monitor_state(NS_IDLE);
    #else
        ns_itm_printf_enable();
    #endif
    ns_lp_printf("Hello!\n");

    ns_audio_init(&audio_config);
    ns_peripheral_button_init(&button_config_nnsp);

    // initialize neural nets controller
    s2iCntrlClass_init(&cntrl_inst);

#ifdef DEF_ACC32BIT_OPT
    ns_lp_printf("You are using \"32bit\" accumulator.\n");
#else
    ns_lp_printf("You are using \"64bit\" accumulator.\n");
#endif
    ns_lp_printf("The estimate time per inference (10 ms data) will be ...\n");
    arm_test_s2i(
        &cntrl_inst, 
        g_in16AudioDataBuffer);
    test_fft();
    test_feat();
    // reset all internal states
    s2iCntrlClass_reset(&cntrl_inst);

    ns_lp_printf("\nPress button to start!\n");
    while (1) 
    {
        g_audioRecording = false;
        g_intButtonPressed = 0;
        
        ns_deep_sleep();
        
        if ( (g_intButtonPressed == 1) && (!g_audioRecording) ) 
        {
            ns_lp_printf("\nYou'd pressed the button. Program start!\n");
            g_intButtonPressed = 0;
            g_audioRecording = true;
            am_hal_delay_us(10);   
            while (1)
            {   
                ns_set_power_monitor_state(NS_DATA_COLLECTION);
                ns_deep_sleep();

                if (g_audioReady) 
                {
                    // execution of each time frame data
                    s2iCntrlClass_exec(
                        &cntrl_inst,
                        g_in16AudioDataBuffer);
                    g_audioReady = false;
                }
            }
            ns_lp_printf("\nPress button to start!\n");
        }
    } // while(1)
}
