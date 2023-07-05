#ifndef __NNID_CLASS_H__
#define __NNID_CLASS_H__
#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>
typedef struct
{
	int32_t *pt_embd;
	int16_t dim_embd;
	int16_t is_get_corr;
	int16_t thresh_get_corr;
	float thresh_trigger;
	float corr;
}NNID_CLASS;

void nnidClass_get_cos(
	int32_t* pt_nn_est,
	int32_t* pt_embd,
	int16_t len,
	float* corr);

void nnidClass_reset_states(
	NNID_CLASS* pt_nnid);
#ifdef __cplusplus
}
#endif
#endif