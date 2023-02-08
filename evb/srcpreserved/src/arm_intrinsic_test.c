#include <stdint.h>
#include <cmsis_gcc.h>
#include "s2iCntrlClass.h"
// #include "nnCntrlClass.h"
#include "arm_intrinsic_test.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_timer.h"
#include "fft_arm.h"
#include "feature_module.h"
#define NUM_FRAMES_EST 1000
// int test_arm_fft()
// {
//     int32_t output[512 << 1];
//     int32_t input[512+2];
//     arm_fft_init();
//     for (int i = 0; i < 512; i++)
//     {
//         input[i] = i <<15;
//     }
//     arm_fft_exec(output, input);
//     for (int i = 0; i < 257; i++)
//     {
//         ns_lp_printf("%d: (%d, %d)\n", i, 
//                     output[2*i] >> 6, 
//                     output[2*i+1]>>6);
//     }
//     return 0;
// }
int arm_test_s2i(
        void *pt_cntrl_inst_, 
        int16_t *pt_data)
{
    uint32_t elapsed_time;
    s2iCntrlClass *pt_cntrl_inst = (s2iCntrlClass *) pt_cntrl_inst_; 
    
    // reset all internal states
    s2iCntrlClass_reset(pt_cntrl_inst);
    ns_timer_init(0);
    for (int i = 0; i < NUM_FRAMES_EST; i++)
    {
        s2iCntrlClass_exec(pt_cntrl_inst, pt_data);
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_lp_printf("Total: %3.2f ms/inference\n",
                ((float) elapsed_time) / NUM_FRAMES_EST / 1000);
    return 0;
}

uint32_t test_feat()
{
    FeatureClass feat;
    uint32_t elapsed_time;
    int32_t mean[50];
    int32_t stdR[50];
    int16_t input[160];
    int i;

    FeatureClass_construct(&feat, mean, stdR, 15);
    FeatureClass_setDefault(&feat);
    ns_timer_init(0);
    for (i=0; i < NUM_FRAMES_EST; i++)
    {
        FeatureClass_execute(&feat, input);
    }
    elapsed_time = ns_us_ticker_read(0);   
    ns_lp_printf("feat: %3.2f ms/inference\n",
                ((float) elapsed_time) / NUM_FRAMES_EST / 1000);
    return 0;
}

uint32_t test_fft()
{
    int32_t input[512 + 2];
    int32_t output[512 << 1];
    uint32_t elapsed_time;
    int i;
    arm_fft_init();
    ns_timer_init(0);
    for (i = 0; i < NUM_FRAMES_EST; i++)
    {
        arm_fft_exec(output, input);
    }
    elapsed_time = ns_us_ticker_read(0);   
    ns_lp_printf("fft: %3.2f ms/inference\n",
                ((float) elapsed_time) / NUM_FRAMES_EST / 1000);
    return 0;
}

#if 0
int arm_test_nnsp(
        void *pt_cntrl_inst_, 
        int16_t *pt_data,
        int16_t *data_buf)
{
    #if 0
    val64 = 0;
    ns_timer_init(0);
    for (i = 0; i < 10000; i++)
    {
        val32 = __SMLAD(val32, val32, val32);
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_lp_printf("%u\n", elapsed_time);

    val64 = 0;
    ns_timer_init(0);
    for (i = 0; i < 10000; i++)
    {
        val64 = __SMLALD(val32, val32, val64);
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_lp_printf("%u\n", elapsed_time);
    
    val64 = 0;
    ns_timer_init(0);
    for (i = 0; i < 10000; i++)
    {
        val64 += (int64_t) val8 * (int64_t) val8 + (int64_t) val8 * (int64_t) val8; 
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_lp_printf("%u\n", elapsed_time);
    #endif
    nnCntrlClass *pt_cntrl_inst = (nnCntrlClass *) pt_cntrl_inst_;
    int i;
    uint32_t elapsed_time;
    // reset all internal states
    nnCntrlClass_reset(pt_cntrl_inst);
    ns_timer_init(0);
    for (i = 0; i < 1000; i++)
    {
        nnCntrlClass_exec(pt_cntrl_inst, pt_data, data_buf);
    }
    elapsed_time = ns_us_ticker_read(0);    
    ns_lp_printf("%3.5f ms/inference\n", ((float) elapsed_time) / 1000 / 1000) ;
    
    

    return 0;
}
#endif