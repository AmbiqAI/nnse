#include <stdint.h>
#include <cmsis_gcc.h>
#include "s2iCntrlClass.h"
// #include "nnCntrlClass.h"
#include "arm_intrinsic_test.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_timer.h"
#include "fft_arm.h"
#include "feature_module.h"
#include "neural_nets.h"
#include <arm_math.h>
// #include "def_nn3_se.h"
#define NUM_FRAMES_EST 1000

ns_timer_config_t my_tickTimer = {
    .prefix = {0},
    .timer = NS_TIMER_COUNTER,
    .enableInterrupt = false,
};

int arm_test_s2i(
        void *pt_cntrl_inst_, 
        int16_t *pt_data)
{
    uint32_t elapsed_time;
    s2iCntrlClass *pt_cntrl_inst = (s2iCntrlClass *) pt_cntrl_inst_; 
    
    // reset all internal states
    s2iCntrlClass_reset(pt_cntrl_inst);
	ns_timer_init(&my_tickTimer);
    for (int i = 0; i < NUM_FRAMES_EST; i++)
    {
        s2iCntrlClass_exec(pt_cntrl_inst, pt_data);
    }
    elapsed_time = ns_us_ticker_read(&my_tickTimer);    
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

    FeatureClass_construct(&feat, mean, stdR, 15, 40);
    FeatureClass_setDefault(&feat);
	ns_timer_init(&my_tickTimer);
    for (i=0; i < NUM_FRAMES_EST; i++)
    {
        FeatureClass_execute(&feat, input);
    }
    elapsed_time = ns_us_ticker_read(&my_tickTimer);    
    ns_lp_printf("feat: %3.2f ms/inference\n",
                ((float) elapsed_time) / NUM_FRAMES_EST / 1000);
    return 0;
}

uint32_t test_fft_ifft()
{
    int i;
    int32_t input[512 << 1] = {0,13527,7337,13487,11953,9883,7363,5938,10921,7094,9281,25198,12700,12043,6717,1541,-3227,-4707,-1465,-211,-5175,13281,3575,-271,-2353,-5177,-6552,-4137,116,6843,-1786,13308,8192,-2607,-4441,-8891,-10620,-9170,-7295,-780,-11480,-4986,-3277,-18369,-17628,-19165,-16337,-11159,-5499,3827,-682,-70,11413,-2419,343,2257,7203,12253,17033,21704,18123,6996,18127,3260,0,1147,3442,5658,7931,7942,6548,-9662,982,-5640,-10662,-4384,77,4459,7663,6900,8006,-7945,-2194,194,-9007,-2200,79,2019,1198,-3179,-4675,-18730,-21049,-10013,-19896,-12268,-8193,-4668,-4385,-5837,-5297,-9892,-14831,5469,1056,8019,14213,16906,15556,12970,10120,8639,-2829,14747,12505,10393,13473,10691,5558,482,-4569,-2933,-14020,-75,6777,590,5507,3099,0,-3100,-5508,-591,-6778,74,14019,2932,4568,-483,-5559,-10692,-13474,-10394,-12506,-14748,2828,-8640,-10121,-12971,-15557,-16907,-14214,-8020,-1057,-5470,14830,9891,5296,5836,4384,4667,8191,12267,19895,10012,21048,18729,4674,3178,-1199,-2020,-80,2199,9006,-195,2193,7944,-8007,-6901,-7664,-4460,-78,4383,10661,5639,-983,9661,-6549,-7943,-7932,-5659,-3443,-1148,-1,-3261,-18128,-6997,-18124,-21705,-17034,-12254,-7204,-2258,-344,2418,-11414,69,681,-3828,5498,11158,16336,19164,17627,18368,3276,4985,11479,779,7294,9169,10619,8890,4440,2606,-8192,-13309,1785,-6844,-117,4136,6551,5176,2352,270,-3576,-13282,5174,210,1464,4706,3226,-1542,-6718,-12044,-12701,-25199,-9282,-7095,-10922,-5939,-7364,-9884,-11954,-13488,-7338,-13528,-1,13527,7337,13487,11953,9883,7363,5938,10921,7094,9281,25198,12700,12043,6717,1541,-3227,-4707,-1465,-211,-5175,13281,3575,-271,-2353,-5177,-6552,-4137,116,6843,-1786,13308,8192,-2607,-4441,-8891,-10620,-9170,-7295,-780,-11480,-4986,-3277,-18369,-17628,-19165,-16337,-11159,-5499,3827,-682,-70,11413,-2419,343,2257,7203,12253,17033,21704,18123,6996,18127,3260,0,1147,3442,5658,7931,7942,6548,-9662,982,-5640,-10662,-4384,77,4459,7663,6900,8006,-7945,-2194,194,-9007,-2200,79,2019,1198,-3179,-4675,-18730,-21049,-10013,-19896,-12268,-8193,-4668,-4385,-5837,-5297,-9892,-14831,5469,1056,8019,14213,16906,15556,12970,10120,8639,-2829,14747,12505,10393,13473,10691,5558,482,-4569,-2933,-14020,-75,6777,590,5507,3099,0,-3100,-5508,-591,-6778,74,14019,2932,4568,-483,-5559,-10692,-13474,-10394,-12506,-14748,2828,-8640,-10121,-12971,-15557,-16907,-14214,-8020,-1057,-5470,14830,9891,5296,5836,4384,4667,8191,12267,19895,10012,21048,18729,4674,3178,-1199,-2020,-80,2199,9006,-195,2193,7944,-8007,-6901,-7664,-4460,-78,4383,10661,5639,-983,9661,-6549,-7943,-7932,-5659,-3443,-1148,0,-3261,-18128,-6997,-18124,-21705,-17034,-12254,-7204,-2258,-344,2418,-11414,69,681,-3828,5498,11158,16336,19164,17627,18368,3276,4985,11479,779,7294,9169,10619,8890,4440,2606,-8192,-13309,1785,-6844,-117,4136,6551,5176,2352,270,-3576,-13282,5174,210,1464,4706,3226,-1542,-6718,-12044,-12701,-25199,-9282,-7095,-10922,-5939,-7364,-9884,-11954,-13488,-7338,-13528,};
    int32_t output[512 << 1];
    uint32_t 	bitReverseFlag=1;
    arm_rfft_instance_q31 fft_st;
    arm_rfft_instance_q31 ifft_st;
    arm_rfft_init_q31(  &fft_st,
                        LEN_FFT_NNSP, 
                        0, 
                        bitReverseFlag);
    arm_rfft_init_q31(  &ifft_st,
                        LEN_FFT_NNSP, 
                        1, 
                        bitReverseFlag);
    for (i = 0; i < 512; i++)
        input[i] <<= 9;
    arm_rfft_q31(&fft_st, input, output);
    for (i = 0; i < 257; i++)
    {
        if ((i==0) || (i==256))
        {
             ns_lp_printf("%d: %d, %d\n",
                    i, output[2*i], output[i * 2 + 1]);

        }
        else
            ns_lp_printf("%d: %d, %d; %d: %d, %d\n",
                    i, output[2*i], output[i * 2 + 1],
                    (512-i), output[2*(512-i)], output[(512-i) * 2 + 1]);
    }

    ns_lp_printf("\n");

    arm_rfft_q31(&ifft_st, output, input);
    for (i = 0; i < 512; i++)
    {
        ns_lp_printf("%d: %d\n", i, input[i]);
    }

    ns_lp_printf("[");
    for (i = 0; i < 512; i++)
    {
        ns_lp_printf("%d, ", input[i]);
    }
    ns_lp_printf("];\n");
    return 0;
}

uint32_t test_fft()
{
    int32_t input[512 + 2];
    int32_t output[512 << 1];
    uint32_t elapsed_time;
    arm_rfft_instance_q31 fft_st;
    int i;
    
    arm_fft_init(&fft_st, 1);
	ns_timer_init(&my_tickTimer);
    for (i = 0; i < NUM_FRAMES_EST; i++)
    {
        arm_fft_exec(&fft_st, output, input);
    }
    elapsed_time = ns_us_ticker_read(&my_tickTimer);    
    ns_lp_printf("fft: %3.2f ms/inference\n",
                ((float) elapsed_time) / NUM_FRAMES_EST / 1000);
    return 0;
}
