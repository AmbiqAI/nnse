#ifndef __NN_CONTROL_CLASS_H__
#define __NN_CONTROL_CLASS_H__
#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>
#define SAMPLES_FRM_NNCNTRL_CLASS 160
#define MAX_NUM_NN 5

// A collection of NN control paramters
typedef struct {
    // VAD
    int16_t thresh_prob_vad;        // threshold for trigger prob
    int16_t thresh_cnts_vad;         // threshold for number of counts to trigger
    
    // S2I
    int16_t frs_vbufBk_s2i;         // lookback frames when s2i starts
    int16_t thresh_timeout_s2i;     // thres_timeout_s2i
    int16_t thresh_prob_s2i;        // threshold for trigger prob
    int16_t thresh_cnts_s2i;        // threshold for number of counts to trigger

    // KWS
    int16_t frs_vbufBk_kws;         // lookback frames when kws starts
    int16_t thresh_timeout_kws;     // thresh_timeout_kws
    int16_t thresh_prob_kws;        // threshold for trigger prob
    int16_t thresh_cnts_kws;        // threshold for number of counts to trigger

}ParamCntrlClass;

/*
    NN Control class: manipulate the sequence of sevearal NNs, such
    as VAD, KWS, S2I
*/
typedef struct{
    void *pt_seq_cntrl;
    int8_t len_seq_cntrl;
    int8_t current_pos_seq;
    void *pt_nnsp_arry;
    ParamCntrlClass Params;
    uint16_t cnt_timeout_kws;
    uint16_t cnt_timeout_s2i;
    uint16_t cnt_voice_frames_detected;
    uint16_t cnt_voice_frames_not_detected;
}nnCntrlClass;

void nnCntrlClass_init(
    nnCntrlClass *pt_inst,
    void* pt_seq_cntrl,
    int8_t len_seq_cntrl);

void nnCntrlClass_reset(nnCntrlClass *pt_inst);

void nnCntrlClass_exec(
        nnCntrlClass *pt_inst, 
        int16_t *data_fr,
        int16_t *pcmbuf_chunk);

void display_current_status(int8_t nnsp_id);
#ifdef __cplusplus
}
#endif
#endif
