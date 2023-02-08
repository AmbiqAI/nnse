#include <stdint.h>
#include <arm_math.h>           // for fft only
#include "nnsp_identification.h"
#include "feature_module.h"
#include "def_nn0_s2i.h"
#include "def_nn1_vad.h"
#include "def_nn2_kws_galaxy.h"
#include "PcmBufClass.h"
#include "nn_speech.h"
#include "nnCntrlClass.h"
#include "ParamsNNCntrl.h"
#include "ns_ambiqsuite_harness.h"
#include "ns_energy_monitor.h"
#include "neural_nets.h"

#ifdef AUDIODEBUG
#include "SEGGER_RTT.h"
#endif

const char* intents[] = {
	"none",
	"change language",
	"bring",
	"activate",
	"deactivate",
	"increase",
	"decrease",
};

const char* slots[] = {
	"none",
	"washroom",
	"juice",
	"volume",
	"shoes",
	"music",
	"heat",
	"lights",
	"kitchen",
	"newspaper",
	"lamp",
	"chinese",
	"german",
	"korean",
	"english",
	"bedroom",
	"socks",
};

NNSPClass NNSP_INSTS[MAX_NUM_NN]; //10ms

FeatureClass FEAT_INSTS[MAX_NUM_NN];

PcmBufClass pcmbuf_inst;

void nnCntrlClass_init(
        nnCntrlClass *pt_inst,   
        void *pt_seq_cntrl_start,
        int8_t len_seq_cntrl) {
    /*
    nnCntrlClass_init: initialization of nnCntrlClass

        Inputs: 
                *pt_inst        :  instance pointer
                pt_seq_cntrl    : nn sequence
                len_seq_cntrl   : len of the sequence
    */
    int i;
    NNSPClass *pt_nnsp_arry = NNSP_INSTS;
    pt_inst->pt_seq_cntrl = pt_seq_cntrl_start;
    pt_inst->len_seq_cntrl = len_seq_cntrl;
    pt_inst->pt_nnsp_arry= (void*) NNSP_INSTS;

    // VAD
    pt_inst->Params.thresh_prob_vad      = THRESHOLD_MASK_VAD;
    pt_inst->Params.thresh_cnts_vad      = THRESHOLD_CNTS_VAD;
    
    // S2I
    pt_inst->Params.frs_vbufBk_s2i       = FRS_VBUFBK_S2IRNN;
    pt_inst->Params.thresh_timeout_s2i   = TIMEOUT_FRS_S2IRNN;
    pt_inst->Params.thresh_prob_s2i      = THRESHOLD_MASK_S2IRNN;
    pt_inst->Params.thresh_cnts_s2i      = THRESHOLD_CNTS_S2IRNN;

    // KWS_GALGAXY
    pt_inst->Params.frs_vbufBk_kws       = FRS_VBUFBK_KWS;
    pt_inst->Params.thresh_timeout_kws   = TIMEOUT_FRS_KWS;
    pt_inst->Params.thresh_prob_kws      = THRESHOLD_MASK_KWS;
    pt_inst->Params.thresh_cnts_kws      = THRESHOLD_CNTS_KWS;

    PcmBufClass_init(&pcmbuf_inst);
    
    i = 0;
    NNSPClass_init(
            &pt_nnsp_arry[i],
            (void*) &net_s2i,
            (void*) &FEAT_INSTS[i],
            i,
            feature_mean_s2i,
            feature_stdR_s2i,
            &pt_inst->Params.thresh_prob_s2i,
            &pt_inst->Params.thresh_cnts_s2i);

    i = 1;
    NNSPClass_init(
            &pt_nnsp_arry[i],
            (void*) &net_vad,
            (void*) &FEAT_INSTS[i],
            i,
            feature_mean_vad,
            feature_stdR_vad,
            &pt_inst->Params.thresh_prob_vad,
            &pt_inst->Params.thresh_cnts_vad);

    i = 2;
    NNSPClass_init(
            &pt_nnsp_arry[i],
            (void*) &net_kws_galaxy,
            (void*) &FEAT_INSTS[i],
            i,
            feature_mean_kws_galaxy,
            feature_stdR_kws_galaxy,
            &pt_inst->Params.thresh_prob_kws,
            &pt_inst->Params.thresh_cnts_kws);

    pt_inst->current_pos_seq = 0;
    pt_inst->cnt_voice_frames_detected = 0;
    pt_inst->cnt_voice_frames_not_detected = 0;
}

void nnCntrlClass_reset(nnCntrlClass *pt_inst) {
    /*
    Reset the states in nnCntrlClass
        
        Inputs:
                pt_inst:  instance pointer
    */
    int i;
    NNSPClass *pt_nnsp_arry = (NNSPClass*)  pt_inst->pt_nnsp_arry;

    pt_inst->cnt_timeout_kws = 0;
    pt_inst->cnt_timeout_s2i = 0;
    pt_inst->cnt_voice_frames_detected = 0;
    pt_inst->cnt_voice_frames_not_detected = 0;

    for (i = 0; i < num_NNSP_IDS; i++)
    {
        NNSPClass_reset(&pt_nnsp_arry[i]);
    }
    PcmBufClass_reset(&pcmbuf_inst);
}

void nnCntrlClass_exec(
        nnCntrlClass *pt_inst, 
        int16_t *data_fr,
        int16_t *pcmbuf_chunk) {
    /***
    execution of each frame data:
        
        Inputs:
                pt_inst : instance pointer
                indata  : frame data    
    ***/
    NNSP_ID *pt_seq_cntrl = (NNSP_ID*) pt_inst->pt_seq_cntrl;
    int8_t current_nnsp_id = pt_seq_cntrl[pt_inst->current_pos_seq];
    NNSPClass *pt_nnsp = ((NNSPClass*) pt_inst->pt_nnsp_arry) + current_nnsp_id;
    int8_t next_nnsp_id = current_nnsp_id;
    int8_t next_pos_seq;
    int16_t detected = 0;

    PcmBufClass_setData(&pcmbuf_inst, data_fr); // put data to voice buffer

    switch (current_nnsp_id)
    {
        case s2i_id: // s2i case
        ns_set_power_monitor_state(NS_INFERING);

        PcmBufClass_getData(
            &pcmbuf_inst,
            pt_inst->Params.frs_vbufBk_s2i,
            1, 
            pcmbuf_chunk); // fetch input data to NN from voice bufffer

        detected = NNSPClass_exec(pt_nnsp, pcmbuf_chunk);

        pt_inst->cnt_timeout_s2i = (pt_inst->cnt_timeout_s2i + 1) % pt_inst->Params.thresh_timeout_s2i;
        
        if (detected || (pt_inst->cnt_timeout_s2i == (pt_inst->Params.thresh_timeout_s2i-1)))
        {
            if (detected)
            {
                ns_lp_printf("\nDetected: %s, %s, %s.\n",
                        intents[pt_nnsp->outputs[0]],
                        slots[pt_nnsp->outputs[1]],
                        slots[pt_nnsp->outputs[2]]);
            }

            next_pos_seq = (pt_inst->current_pos_seq + 1) % pt_inst->len_seq_cntrl; 
            next_nnsp_id = pt_seq_cntrl[next_pos_seq];

            if ((detected) || (current_nnsp_id != next_nnsp_id))
            {
                pt_inst->cnt_timeout_s2i = 0;
                NNSPClass_reset(pt_nnsp);
            } 
            pt_inst->current_pos_seq = next_pos_seq;
        }
        break;

        case kws_galaxy_id: // kws case

        PcmBufClass_getData(
            &pcmbuf_inst,
            pt_inst->Params.frs_vbufBk_kws,
            1, 
            pcmbuf_chunk); // fetch input data to NN from voice bufffer

        detected = NNSPClass_exec(pt_nnsp, pcmbuf_chunk);

        pt_inst->cnt_timeout_kws = (pt_inst->cnt_timeout_kws + 1) % pt_inst->Params.thresh_timeout_kws;
        
        if (detected || (pt_inst->cnt_timeout_kws == (pt_inst->Params.thresh_timeout_kws-1)))
        {
            if (detected)
            {
                next_pos_seq = (pt_inst->current_pos_seq + 1) % pt_inst->len_seq_cntrl; 
                next_nnsp_id = pt_seq_cntrl[next_pos_seq];
                ns_lp_printf("\nDetected: Hi Galaxy\n");
            }
            else
            {
                next_pos_seq = (pt_inst->current_pos_seq - 1) % pt_inst->len_seq_cntrl; 
                next_pos_seq = (next_pos_seq >= 0) ? next_pos_seq : next_pos_seq + pt_inst->len_seq_cntrl;
                next_nnsp_id = pt_seq_cntrl[next_pos_seq];
            }

            if ((detected) || (current_nnsp_id != next_nnsp_id))
            {
                pt_inst->cnt_timeout_kws = 0;
                NNSPClass_reset(pt_nnsp);
            } 
            pt_inst->current_pos_seq = next_pos_seq;
        }
        break;

        case vad_id: // vad case
        ns_set_power_monitor_state(NS_FEATURE_EXTRACTION);

        PcmBufClass_getData(
            &pcmbuf_inst,
            0,
            1, 
            pcmbuf_chunk); // fetch input data to NN from voice bufffer

        detected = NNSPClass_exec(pt_nnsp, pcmbuf_chunk);

        if (detected)
        {
            ns_lp_printf("\nVoice detected!!\n");
            next_pos_seq = (pt_inst->current_pos_seq + 1) % pt_inst->len_seq_cntrl;
            next_nnsp_id = pt_seq_cntrl[next_pos_seq];

            if ((detected) || (current_nnsp_id != next_nnsp_id))
            {
                NNSPClass_reset(pt_nnsp);
            } 
            pt_inst->current_pos_seq = next_pos_seq;
        }
        break;
    }
    if (current_nnsp_id != next_nnsp_id)
        display_current_status(next_nnsp_id);
}


void nnCntrlClass_exec_ignoreme(
        nnCntrlClass *pt_inst, 
        int16_t *data_fr,
        int16_t *pcmbuf_chunk) {
    /***
    execution of each frame data:
        
        Inputs:
                pt_inst : instance pointer
                indata  : frame data    
    ***/
    NNSP_ID *pt_seq_cntrl = (NNSP_ID*) pt_inst->pt_seq_cntrl;
    int8_t current_nnsp_id = pt_seq_cntrl[pt_inst->current_pos_seq];
    NNSPClass *pt_nnsp = ((NNSPClass*) pt_inst->pt_nnsp_arry) + current_nnsp_id;
    NNSPClass *pt_vad = ((NNSPClass*) pt_inst->pt_nnsp_arry) + vad_id;
    int8_t next_nnsp_id = current_nnsp_id;
    int8_t next_pos_seq;
    int16_t detected = 0;

    PcmBufClass_setData(&pcmbuf_inst, data_fr); // put data to voice buffer

    switch (current_nnsp_id)
    {
        case s2i_id: // s2i case
        // VAD first
        // PcmBufClass_getData(
        //     &pcmbuf_inst,
        //     0,
        //     1, 
        //     pcmbuf_chunk); // fetch input data to NN from voice bufffer

        // detected = NNSPClass_exec(pt_vad, pcmbuf_chunk);
        // if (detected == false) {
        //     pt_inst->cnt_voice_frames_not_detected++;
        //     ns_lp_printf(".");
        // } else {
        //     pt_inst->cnt_voice_frames_not_detected = 0;
        // }

        ns_set_power_monitor_state(NS_INFERING);

        PcmBufClass_getData(
            &pcmbuf_inst,
            pt_inst->Params.frs_vbufBk_s2i,
            1, 
            pcmbuf_chunk); // fetch input data to NN from voice bufffer

        detected = NNSPClass_exec(pt_nnsp, pcmbuf_chunk);

        pt_inst->cnt_timeout_s2i = (pt_inst->cnt_timeout_s2i + 1) % pt_inst->Params.thresh_timeout_s2i;
        
        if (detected || 
            (pt_inst->cnt_timeout_s2i == (pt_inst->Params.thresh_timeout_s2i-1)) ||
            (pt_inst->cnt_voice_frames_not_detected > 5))
        {
            if (detected)
            {
                ns_lp_printf("\nDetected: %s, %s, %s.\n",
                        intents[pt_nnsp->outputs[0]],
                        slots[pt_nnsp->outputs[1]],
                        slots[pt_nnsp->outputs[2]]);
            }

            next_pos_seq = (pt_inst->current_pos_seq + 1) % pt_inst->len_seq_cntrl; 
            next_nnsp_id = pt_seq_cntrl[next_pos_seq];

            if ((detected) || (current_nnsp_id != next_nnsp_id))
            {
                pt_inst->cnt_timeout_s2i = 0;
                pt_inst->cnt_voice_frames_not_detected = 0;
                NNSPClass_reset(pt_nnsp);
            } 
            pt_inst->current_pos_seq = next_pos_seq;
        }
        break;

        case kws_galaxy_id: // kws case

        PcmBufClass_getData(
            &pcmbuf_inst,
            pt_inst->Params.frs_vbufBk_kws,
            1, 
            pcmbuf_chunk); // fetch input data to NN from voice bufffer

        detected = NNSPClass_exec(pt_nnsp, pcmbuf_chunk);

        pt_inst->cnt_timeout_kws = (pt_inst->cnt_timeout_kws + 1) % pt_inst->Params.thresh_timeout_kws;
        
        if (detected || (pt_inst->cnt_timeout_kws == (pt_inst->Params.thresh_timeout_kws-1)))
        {
            if (detected)
            {
                next_pos_seq = (pt_inst->current_pos_seq + 1) % pt_inst->len_seq_cntrl; 
                next_nnsp_id = pt_seq_cntrl[next_pos_seq];
                ns_lp_printf("\nDetected: Hi Galaxy\n");
            }
            else
            {
                next_pos_seq = (pt_inst->current_pos_seq - 1) % pt_inst->len_seq_cntrl; 
                next_pos_seq = (next_pos_seq >= 0) ? next_pos_seq : next_pos_seq + pt_inst->len_seq_cntrl;
                next_nnsp_id = pt_seq_cntrl[next_pos_seq];
            }

            if ((detected) || (current_nnsp_id != next_nnsp_id))
            {
                pt_inst->cnt_timeout_kws = 0;
                NNSPClass_reset(pt_nnsp);
            } 
            pt_inst->current_pos_seq = next_pos_seq;
        }
        break;

        case vad_id: // vad case
        ns_set_power_monitor_state(NS_FEATURE_EXTRACTION);

        PcmBufClass_getData(
            &pcmbuf_inst,
            0,
            1, 
            pcmbuf_chunk); // fetch input data to NN from voice bufffer

        detected = NNSPClass_exec(pt_nnsp, pcmbuf_chunk);

        if (detected)
        {
            if (pt_inst->cnt_voice_frames_detected < FRS_VBUFBK_S2IRNN) {
                pt_inst->cnt_voice_frames_detected++;
            } else {
                ns_lp_printf("\nVoice detected for %d frames!!\n", FRS_VBUFBK_S2IRNN);
                next_pos_seq = (pt_inst->current_pos_seq + 1) % pt_inst->len_seq_cntrl;
                next_nnsp_id = pt_seq_cntrl[next_pos_seq];
                pt_inst->cnt_voice_frames_detected = 0;

                if ((detected) || (current_nnsp_id != next_nnsp_id))
                {
                    NNSPClass_reset(pt_nnsp);
                } 
                pt_inst->current_pos_seq = next_pos_seq;
            }

        } else {
            pt_inst->cnt_voice_frames_detected = 0;
        }

        break;
    }
    if (current_nnsp_id != next_nnsp_id)
        display_current_status(next_nnsp_id);
}

void display_current_status(int8_t nnsp_id) {
    /*
    Display current running NN
    */
    switch (nnsp_id)
    {
        case s2i_id:
        ns_lp_printf("--------------In S2I phase--------------\n");
        break;

        case kws_galaxy_id:
        ns_lp_printf("--------------In KWS phase--------------\n");
        break;

        case vad_id:
        ns_lp_printf("--------------In VAD phase--------------\n");
        break;
    }
}
