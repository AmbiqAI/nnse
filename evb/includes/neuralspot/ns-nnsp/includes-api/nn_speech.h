/* 
This module integrates neural nets, feature extraction & post processing
*/
#ifndef __NN_SPEECH_H__
#define __NN_SPEECH_H__
#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>

typedef struct
{
	char nn_id;
	void* pt_net;
	void* pt_feat;
	int8_t slides;
	int16_t trigger;
	int16_t *pt_thresh_prob;
	int16_t counts_category[8];
	int16_t *pt_th_count_trigger;
	int16_t num_dnsmpl;
	int16_t outputs[3];
	int16_t argmax_last;
	int16_t *pt_se_out;
}NNSPClass;

int NNSPClass_init(
	NNSPClass* pt_inst,
	void* pt_net,  // NeuralNetClass
	void* pt_feat, // featureModule
	char nn_id,
	const int32_t* pt_mean,
	const int32_t* pt_stdR,
	int16_t* pt_thresh_prob,
	int16_t* pt_th_count_trigger,
	int16_t num_mfltrBank,
	int16_t num_dnsmpl);

int NNSPClass_reset(NNSPClass* pt_inst);

int16_t NNSPClass_exec(
		NNSPClass* pt_inst,
	 	int16_t* rawPCM);

void my_argmax(int32_t *vec, int len, int16_t *Imax);

int32_t compute_pwr2(int32_t input);

int32_t ceiling(int32_t input);

void binary_post_proc(
	NNSPClass* pt_inst,
	int32_t* pt_nn_est, 
	int16_t *pt_trigger);

void s2i_post_proc(
	NNSPClass* pt_inst,
	int32_t* pt_nn_est,
	int16_t *pt_trigger);

void se_post_proc(
	NNSPClass* pt_inst,
	int16_t *pt_nn_est,
	int16_t *pt_se_out);

#ifdef __cplusplus
}
#endif
#endif