#include <stdint.h>
#include "feature_module.h"
#include "def_nn0_s2i.h"
#include "PcmBufClass.h"
#include "nn_speech.h"
#include "s2iCntrlClass.h"
#include "ParamsNNCntrl.h"
#include "ns_ambiqsuite_harness.h"
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

NNSPClass NNSP_INST; //10ms

FeatureClass FEAT_INST;

PcmBufClass PCMBUF_INST;

void s2iCntrlClass_init(
        s2iCntrlClass *pt_inst) {
    /*
    s2iCntrlClass_init: initialization of nnCntrlClass

        Inputs: 
                *pt_inst        :  instance pointer
    */
    pt_inst->pt_nnsp= (void*) &NNSP_INST;

    // S2I
    pt_inst->Params.frs_vbufBk_s2i       = FRS_VBUFBK_S2IRNN;
    pt_inst->Params.thresh_prob_s2i      = THRESHOLD_MASK_S2IRNN;
    pt_inst->Params.thresh_cnts_s2i      = THRESHOLD_CNTS_S2IRNN;

    PcmBufClass_init(&PCMBUF_INST);

    NNSPClass_init(
        (NNSPClass*) pt_inst->pt_nnsp,
        (void*) &net_s2i,
        (void*) &FEAT_INST,
        0,
        feature_mean_s2i,
        feature_stdR_s2i,
        &pt_inst->Params.thresh_prob_s2i,
        &pt_inst->Params.thresh_cnts_s2i);
}

void s2iCntrlClass_reset(s2iCntrlClass *pt_inst) {
    /*
    Reset the states in s2iCntrlClass
        
        Inputs:
                pt_inst:  instance pointer
    */
    PcmBufClass_reset(&PCMBUF_INST);

    NNSPClass_reset((NNSPClass*) pt_inst->pt_nnsp);

}

void s2iCntrlClass_exec(
        s2iCntrlClass *pt_inst, 
        int16_t *data_fr) {
    /*
    execution of each frame data:
        
        Inputs:
                pt_inst : instance pointer
                indata  : frame data    
    */
    static int16_t pcmbuf_chunk[SAMPLES_FRM_NNCNTRL_CLASS];
    NNSPClass *pt_nnsp = (NNSPClass*) pt_inst->pt_nnsp;
    int16_t detected = 0;

    PcmBufClass_setData(&PCMBUF_INST, data_fr); // put data to voice buffer
    PcmBufClass_getData(
            &PCMBUF_INST,
            0,
            1, 
            pcmbuf_chunk); // fetch input data to NN from voice bufffer

    detected = NNSPClass_exec(pt_nnsp, pcmbuf_chunk);
    if (detected)
    {
        ns_lp_printf("\nDetected: %s, %s, %s.\n",
                intents[pt_nnsp->outputs[0]],
                slots[pt_nnsp->outputs[1]],
                slots[pt_nnsp->outputs[2]]);
        NNSPClass_reset(pt_nnsp);
    }
}
