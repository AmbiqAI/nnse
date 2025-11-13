#include "spectrogram_module.h"
#include "ambiq_stdint.h"
#include "ambiq_nnsp_const.h"
#include "ambiq_nnsp_debug.h"
#include "minmax.h"

#if ARM_FFT == 0
#include "fft.h"
#include "kiss_fftr.h"
#include "kiss_fft.h"
#include <stdio.h>
#else
#include "fft_arm.h"
#endif
static int32_t glob_fft_buf[1026];
static int32_t glob_spec[1026];

int16_t dataBuffer[LEN_FFT_NNSP];
int32_t odataBuffer[LEN_FFT_NNSP];


int stftModule_construct(
    stftModule *ps, int16_t len_win, int16_t hopsize, int16_t fftsize,
    const int16_t *pt_stft_win_coeff) {
    ps->spec = glob_spec;
    ps->dataBuffer = dataBuffer;
    ps->odataBuffer = odataBuffer;
    ps->len_win = len_win;
    ps->hop = hopsize;
    ps->len_fft = fftsize;
    ps->window = pt_stft_win_coeff;
#if ARM_FFT == 1
    arm_fft_init(&ps->fft_st, 0, fftsize);
    arm_fft_init(&ps->ifft_st, 1, fftsize);
#else
    printf("Allocating kiss_fftr configs for len_fft: %d\n", fftsize);
    ps->fft_st = (void *)kiss_fftr_alloc(fftsize, 0, NULL, NULL); // forward
    if (!ps->fft_st) {
        fprintf(stderr, "kiss_fftr_alloc (forward) failed\n");
        return 1;
    }
    ps->ifft_st = (void *)kiss_fftr_alloc(fftsize, 1, NULL, NULL); // inverse
    if (!ps->ifft_st) {
        fprintf(stderr, "kiss_fftr_alloc (inverse) failed\n");
        free(ps->fft_st);
        return 1;
    }
#endif
    return 0;
}
int stftModule_destruct(stftModule *ps) {
 
    // free(icfg);
#if ARM_FFT == 0
    if (ps->fft_st) free(ps->fft_st);
    if (ps->ifft_st) free(ps->ifft_st);
#endif
    return 0;
}
int stftModule_setDefault(stftModule *ps) {
    for (int i = 0; i < ps->len_win; i++) {
        ps->dataBuffer[i] = 0;
        ps->odataBuffer[i] = 0;
    }
    return 0;
}

#if ARM_FFT == 0
void spec2pspec(
        int32_t *y,
        int32_t *x,
        int len,
        int16_t qbit_in) {
    int i;
    int64_t tmp;
    int rshift = (qbit_in << 1) - 15;
    for (i = 0; i < len; i++) {
        tmp = (int64_t)x[2 * i] * (int64_t)x[2 * i] + (int64_t)x[2 * i + 1] * (int64_t)x[2 * i + 1];
        tmp = tmp >> rshift;
        tmp = MIN(MAX(tmp, INT32_MIN), INT32_MAX);
        
        y[i] = (int32_t) tmp;
    }
}

int stftModule_analyze(
        stftModule *ps,
        int16_t *fft_in_q16, // q15
        int32_t *spec,       // q21
        int16_t *pt_qbit_out) 
{
    int i;
    int32_t tmp;
    kiss_fftr_state_T *pt_fft = (kiss_fftr_state_T *) ps->fft_st;

    for (i = 0; i < (ps->len_win - ps->hop); i++)
        ps->dataBuffer[i] = ps->dataBuffer[i + ps->hop];

    tmp = ps->len_win - ps->hop;
    for (i = 0; i < ps->hop; i++)
        ps->dataBuffer[i + tmp] = fft_in_q16[i];

    for (i = 0; i < ps->len_win; i++) {
        tmp = (int32_t)ps->window[i] * (int32_t)ps->dataBuffer[i];
        glob_fft_buf[i] = tmp; // Q30
    }

    for (i = 0; i < (ps->len_fft - ps->len_win); i++) {
        glob_fft_buf[i + ps->len_win] = 0;
    }

    kiss_fftr(
        (kiss_fftr_cfg) ps->fft_st ,
        glob_fft_buf,  // fft_in,  Q30    
        (kiss_fft_cpx*) spec); // ff1t_out, Q21
   
    if (ps->len_fft == 512)
        *pt_qbit_out = 21;
    else
        *pt_qbit_out = 22;

    return 0;
}
int stftModule_synthesize(
    stftModule *ps,
    int32_t *spec,   // Q21
    int16_t *output) // Q15
{
    int i;
    int64_t tmp64;
    int32_t *pt_out;
    kiss_fftr_state_T *pt_ifft = (kiss_fftr_state_T *) ps->ifft_st;

    kiss_fftri(
        pt_ifft,
        (kiss_fft_cpx*) spec,  // Q21
        glob_fft_buf); // Q21

    for (i = 0; i < ps->len_win; i++) {
        
        int64_t sig = (int64_t) glob_fft_buf[i];
        tmp64 = ((int64_t)ps->window[i]) * sig; // Q(21+15=36)
        tmp64 >>= 21; // Q15
        tmp64 = (int64_t)ps->odataBuffer[i] + (int64_t)tmp64;
        tmp64 = MIN(MAX(tmp64, INT32_MIN), INT32_MAX);
        ps->odataBuffer[i] = (int32_t)tmp64;
    }

    for (i = 0; i < ps->hop; i++)
        output[i] = (int16_t)MIN(MAX(ps->odataBuffer[i], INT16_MIN), INT16_MAX);

    for (i = 0; i < ps->len_win - ps->hop; i++) {
        ps->odataBuffer[i] = ps->odataBuffer[i + ps->hop];
    }

    pt_out = ps->odataBuffer + ps->len_win - ps->hop;
    for (i = 0; i < ps->hop; i++) {
        pt_out[i] = 0;
    }

    return 0;
}

#else
/*
        stftModule_analyze_arm: stft analysis
        overlap-and-add approach
*/
#if ARM_OPTIMIZED == 1
void spec2pspec_arm(
    int32_t *pspec, // q15
    int32_t *spec,  // q21
    int len, int16_t qbit_in) {
    int i;
    int64_t acc, tmp_t;
    int32_t *pt_spec = spec;
    int rshift = (qbit_in << 1) - 15;
    
    for (i = 0; i < len; i++) {
        acc = 0;
        tmp_t = (int64_t)*pt_spec++;
        acc = tmp_t * tmp_t;
        tmp_t = (int64_t)*pt_spec++;
        acc += tmp_t * tmp_t; // TODO: check if this is correct
        pspec[i] = (int32_t)MIN(MAX(acc >> rshift, INT32_MIN), INT32_MAX);
    }
    
    
    // ns_printf("spec\n");
    // for (i = 0; i < len*2; i++) {
    //     ns_printf("%d ", spec[i]);
    // }
    // ns_printf("\n");
}
int stftModule_analyze_arm(
    void *ps_t,
    int16_t *fft_in_q16, // q15
    int32_t *spec,       // q21
    int16_t fftsize, int16_t *pt_qbit_out) {
    int i;
    int32_t tmp;
    stftModule *ps = (stftModule *)ps_t;

    for (i = 0; i < (ps->len_win - ps->hop); i++)
        ps->dataBuffer[i] = ps->dataBuffer[i + ps->hop];

    tmp = ps->len_win - ps->hop;
    for (i = 0; i < ps->hop; i++)
        ps->dataBuffer[i + tmp] = fft_in_q16[i];

    for (i = 0; i < ps->len_win; i++) {
        tmp = (int32_t)ps->window[i] * (int32_t)ps->dataBuffer[i];
        glob_fft_buf[i] = tmp; // Q30
    }

    for (i = 0; i < (ps->len_fft - ps->len_win); i++) {
        glob_fft_buf[i + ps->len_win] = 0;
    }

    arm_fft_exec(
        &ps->fft_st,
        spec,          // fft_out, Q21
        glob_fft_buf); // fft_in,  Q30    

    if (fftsize == 512)
        *pt_qbit_out = 21;
    else
        *pt_qbit_out = 22;
    return 0;
}

/*
        stftModule_synthesize_arm: inverse stft synthesization
        overlap-and-add approach
*/
int stftModule_synthesize_arm(
    void *ps_t,
    int32_t *spec,   // Q21
    int16_t *output) // Q15
{
    int i;
    int64_t tmp64;
    stftModule *ps = (stftModule *)ps_t;
    int32_t *pt_out;

    arm_rfft_q31(
        &ps->ifft_st,
        spec,          // Q21
        glob_fft_buf); // Q21

    for (i = 0; i < ps->len_win; i++) {
        tmp64 = ((int64_t)ps->window[i]) * (int64_t)glob_fft_buf[i];
        tmp64 >>= 21;
        tmp64 = (int64_t)ps->odataBuffer[i] + (int64_t)tmp64;
        tmp64 = MIN(MAX(tmp64, INT32_MIN), INT32_MAX);
        ps->odataBuffer[i] = (int32_t)tmp64;
    }

    for (i = 0; i < ps->hop; i++)
        output[i] = (int16_t)MIN(MAX(ps->odataBuffer[i], INT16_MIN), INT16_MAX);

    for (i = 0; i < ps->len_win - ps->hop; i++) {
        ps->odataBuffer[i] = ps->odataBuffer[i + ps->hop];
    }

    pt_out = ps->odataBuffer + ps->len_win - ps->hop;
    for (i = 0; i < ps->hop; i++) {
        pt_out[i] = 0;
    }

    return 0;
}
#elif ARM_OPTIMIZED == 3
#include "basic_mve.h"
#include <arm_mve.h>
void spec2pspec_arm(
    int32_t *pspec, // q15
    int32_t *spec,  // q21
    int len, int16_t qbit_in) 
{
    int i;
    int64_t acc; //, tmp_t;
    int32_t *pt_spec = spec;
    int rshift = (qbit_in << 1) - 15;
    int32x4_t m1;

    for (i = 0; i < len; i++) {
        mve_pred16_t p = vctp32q(2); // Only first 2 lanes active
        m1 = vldrwq_z_s32(pt_spec, p); // 17= (1 << 4) + (1 << 0)
        pt_spec += 2;
        acc= vmlaldavq_s32(m1, m1);
        pspec[i] = (int32_t)MIN(MAX(acc >> rshift, INT32_MIN), INT32_MAX);
    }

    // ns_printf("spec\n");
    // for (i = 0; i < len*2; i++) {
    //     ns_printf("%d ", spec[i]);
    // }
    // ns_printf("\n");
}
int stftModule_analyze_arm(
    void *ps_t,
    int16_t *fft_in_q16, // q15
    int32_t *spec,       // q21
    int16_t fftsize, int16_t *pt_qbit_out) {

    stftModule *ps = (stftModule *)ps_t;

    move_data_16b(
        ps->dataBuffer+ps->hop,
        ps->dataBuffer,
        ps->len_win - ps->hop);

    move_data_16b(
        fft_in_q16,
        ps->dataBuffer+ps->len_win - ps->hop,
        ps->hop);

    vec16_vec16_mul_32b(
        glob_fft_buf,
        (int16_t*) ps->window,
        ps->dataBuffer,
        ps->len_win);

    set_zero_32b(
        glob_fft_buf+ps->len_win,
        ps->len_fft - ps->len_win);

    arm_fft_exec(
        &ps->fft_st,
        spec,          // fft_out, Q21
        glob_fft_buf); // fft_in,  Q30

    if (fftsize == 512)
        *pt_qbit_out = 21;
    else
        *pt_qbit_out = 22;
    return 0;
}

/*
        stftModule_synthesize_arm: inverse stft synthesization
        overlap-and-add approach
*/
int stftModule_synthesize_arm(
    void *ps_t,
    int32_t *spec,   // Q21
    int16_t *output) // Q15
{
    int i;
    int64_t tmp64;
    stftModule *ps = (stftModule *)ps_t;
    int32_t *pt_out;
    arm_rfft_q31(
        &ps->ifft_st,
        spec,          // Q21
        glob_fft_buf); // Q21

    for (i = 0; i < ps->len_win; i++) {
        tmp64 = ((int64_t)ps->window[i]) * (int64_t)glob_fft_buf[i];
        tmp64 >>= 21;
        tmp64 = (int64_t)ps->odataBuffer[i] + (int64_t)tmp64;
        tmp64 = MIN(MAX(tmp64, INT32_MIN), INT32_MAX);
        ps->odataBuffer[i] = (int32_t)tmp64;
    }

    // for (i = 0; i < ps->hop; i++)
    //     output[i] = (int16_t)MIN(MAX(ps->odataBuffer[i], INT16_MIN), INT16_MAX);
    int32x4_t *pt_tt = (int32x4_t*) ps->odataBuffer;
    int32x4_t Mv = {INT16_MAX,INT16_MAX,INT16_MAX,INT16_MAX};
    int32x4_t mv = {INT16_MIN,INT16_MIN,INT16_MIN,INT16_MIN};
    for (i= 0; i < ps->hop >> 2; i++)
    {
        int32x4_t tt = *pt_tt;
        pt_tt+=1;
        tt = vminq_s32(Mv, vmaxq_s32(mv, tt));
        vstrhq_s32(output, tt);
        output+=4;
    }

    // for (i = 0; i < ps->len_win - ps->hop; i++) {
    //     ps->odataBuffer[i] = ps->odataBuffer[i + ps->hop];
    // }
    move_data_16b(
        (int16_t*) (ps->odataBuffer + ps->hop),
        (int16_t*) ps->odataBuffer,
        (ps->len_win - ps->hop) << 1 );

    pt_out = ps->odataBuffer + ps->len_win - ps->hop;
    // for (i = 0; i < ps->hop; i++) {
    //     pt_out[i] = 0;
    // }
    set_zero_32b(pt_out, ps->hop);
    return 0;
}

#endif

#endif
