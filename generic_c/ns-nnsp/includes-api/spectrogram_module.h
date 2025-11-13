#ifndef __SPECTROGRM_MODULE_H__
#define __SPECTROGRM_MODULE_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include "ambiq_nnsp_debug.h"
#if ARM_FFT == 1
    #include <arm_math.h>
#endif
#include "ambiq_nnsp_const.h"
#if ARM_FFT == 0
    #include "fft.h"
    #include "kiss_fftr.h"
    #include "kiss_fft.h"
#endif

typedef struct kiss_fftr_state{
    kiss_fft_cfg substate;
    kiss_fft_cpx * tmpbuf;
    kiss_fft_cpx * super_twiddles;
#ifdef USE_SIMD
    void * pad;
#endif
}kiss_fftr_state_T;

typedef struct {
    int16_t len_win;
    int16_t hop;
    int16_t len_fft;
    int16_t *dataBuffer;
    int32_t *odataBuffer;
    const int16_t *window;
#if ARM_FFT == 1
    arm_rfft_instance_q31 fft_st;
    arm_rfft_instance_q31 ifft_st;
#else
    void *fft_st;
    void *ifft_st;
    // kiss_fftr_cfg icfg;
#endif
    int32_t *spec;
} stftModule;

int stftModule_construct(
    stftModule *ps, int16_t len_win, int16_t hopsize, int16_t fftsize,
    const int16_t *pt_stft_win_coeff);

int stftModule_setDefault(stftModule *ps);
int stftModule_destruct(stftModule *ps);
#if ARM_FFT == 0
void spec2pspec(
        int32_t *y,
        int32_t *x,
        int len,
        int16_t qbit_in);

int stftModule_analyze(
        stftModule *ps,
        int16_t *fft_in_q16, // q15
        int32_t *spec,       // q21
        int16_t *pt_qbit_out);
int stftModule_synthesize(
    stftModule *ps,
    int32_t *spec_t,   // Q21
    int16_t *output); // Q15
#else

void spec2pspec_arm(
    int32_t *pspec, // q15
    int32_t *spec,  // q21
    int len, int16_t qbit_in);

int stftModule_analyze_arm(
    void *ps_t,
    int16_t *fft_in_q16, // q15
    int32_t *spec,       // q21
    int16_t fftsize, int16_t *pt_qbit_out);

int stftModule_synthesize_arm(void *ps_t, int32_t *spec, int16_t *output);
#endif

#ifdef __cplusplus
}
#endif
#endif
