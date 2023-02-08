#ifndef __S2I_CONTROL_CLASS_H__
#define __S2I_CONTROL_CLASS_H__
#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>
#define SAMPLES_FRM_NNCNTRL_CLASS 160
#define MAX_NUM_NN 5

// A collection of NN control paramters
typedef struct {
    // S2I
    int16_t frs_vbufBk_s2i;         // lookback frames when s2i starts
    int16_t thresh_prob_s2i;        // threshold for trigger prob
    int16_t thresh_cnts_s2i;        // threshold for number of counts to trigger
}ParamCntrlClass;

/*
    NN Control class: manipulate the sequence of sevearal NNs, such
    as VAD, KWS, S2I
*/
typedef struct{
    void *pt_nnsp;
    ParamCntrlClass Params;
}s2iCntrlClass;

void s2iCntrlClass_init(
    s2iCntrlClass *pt_inst);

void s2iCntrlClass_reset(s2iCntrlClass *pt_inst);

void s2iCntrlClass_exec(
        s2iCntrlClass *pt_inst, 
        int16_t *data_fr);
#ifdef __cplusplus
}
#endif
#endif
