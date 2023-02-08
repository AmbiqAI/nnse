#include <stdint.h>
#include "nnCntrlClass.h"
#include "ambiq_nnsp_const.h"
#include "nnsp_identification.h"
#include "am_util_stdio.h"
#include "ns_peripherals_button.h"
#include "ns_peripherals_power.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_audio.h"
// #include "arm_intrinsic_test.h"
#include "ns_energy_monitor.h"

#define NUM_CHANNELS 1
int16_t pcmbuf_chunk[LEN_STFT_HOP];

int volatile g_intButtonPressed = 0;
///Button Peripheral Config Struct
ns_button_config_t button_config_nnsp = {
    .button_0_enable = true,
    .button_1_enable = false,
    .button_0_flag = &g_intButtonPressed,
    .button_1_flag = NULL
};

/// Set by app when it wants to start recording, used by callback
bool static g_audioRecording = false;

/// Set by callback when audio buffer has been copied, cleared by
/// app when the buffer has been consumed.
bool static g_audioReady = false;
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
        (uint32_t *)am_hal_audadc_dma_get_buffer(config->audioSystemHandle);

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
    .audioBuffer = (void *)&g_in16AudioDataBuffer,
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

int main(void)
{
    nnCntrlClass cntrl_inst;

    // NNSP_ID nn_seq[]={vad_id, kws_galaxy_id, s2i_id};
    // int8_t len_nn_seq = 3;
    NNSP_ID nn_seq[]={vad_id, s2i_id};
    int8_t len_nn_seq = 2;

    NNSP_ID *pt_seq_cntrl;
    int8_t current_nnsp_id;
    g_audioRecording = false;

    ns_itm_printf_enable();
    
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();

    //
    // Initialize the printf interface for ITM output
    //
    ns_debug_printf_enable();
	ns_power_config(&ns_audio_default);			
    ns_peripheral_button_init(&button_config_nnsp);
    ns_audio_init(&audio_config);

    // initialize neural nets controller
    nnCntrlClass_init(&cntrl_inst, (void*) nn_seq, len_nn_seq);
    pt_seq_cntrl = (NNSP_ID*) cntrl_inst.pt_seq_cntrl;
    current_nnsp_id = pt_seq_cntrl[cntrl_inst.current_pos_seq];

    // reset all internal states
    nnCntrlClass_reset(&cntrl_inst);

    #ifdef DEF_ACC32BIT_OPT
    ns_lp_printf("You are using 32bit accumulator.\n");
#else
    ns_lp_printf("You are using 64bit accumulator.\n");
#endif

    ns_lp_printf("\nPress button to start!\n");

    while (1) 
    {
        g_audioRecording = false;
        g_intButtonPressed = 0;
        
        ns_deep_sleep();
        
        if ( (g_intButtonPressed == 1) && (!g_audioRecording) ) 
        {
            ns_lp_printf("\nYou've pressed the button. Program start!\n");
            display_current_status(current_nnsp_id);
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
                    nnCntrlClass_exec(
                        &cntrl_inst, 
                        g_in16AudioDataBuffer,
                        pcmbuf_chunk);

                    g_audioReady = false;
                }
            }
        }
    } // while(1)
}
