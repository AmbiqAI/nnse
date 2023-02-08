#ifndef __ARM_INTRINSIC_H__
#define __ARM_INTRINSIC_H__
#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>
int arm_test_se();
int arm_test_s2i(
    void *pt_cntrl_inst, 
    int16_t *pt_data);
uint32_t test_feat();
uint32_t test_fft_ifft();
uint32_t test_fft();
#ifdef __cplusplus
}
#endif
#endif