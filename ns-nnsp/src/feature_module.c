#include "ambiq_stdint.h"
#include "spectrogram_module.h"
#include "feature_module.h"
#include "minmax.h"
#include "ambiq_nnsp_const.h"
#include "melSpecProc.h"
#include "fixlog10.h"
#include "ambiq_nnsp_debug.h"
#include "ns_ambiqsuite_harness.h"
#define LOG10_2POW_N15_Q15 (-147963)
int32_t GLOBAL_PSPEC[512];
extern const int16_t mfltrBank_coeff_nfilt72[];
extern const int16_t mfltrBank_coeff_nfilt40[];
void FeatureClass_construct(
		FeatureClass* ps,
		const int32_t* norm_mean,
		const int32_t* norm_stdR,
		int8_t qbit_output,
		int16_t num_mfltrBank)
{
	stftModule_construct(&ps->state_stftModule);
	ps->pt_norm_mean = norm_mean;
	ps->pt_norm_stdR = norm_stdR;
	ps->num_context = NUM_FEATURE_CONTEXT;
	ps->dim_feat = num_mfltrBank;
	ps->qbit_output = qbit_output;
	ps->num_mfltrBank = num_mfltrBank;
	if (ps->num_mfltrBank==72)
		ps->p_melBanks = mfltrBank_coeff_nfilt72;
	else if (ps->num_mfltrBank==40)
		ps->p_melBanks = mfltrBank_coeff_nfilt40;
}

void FeatureClass_setDefault(FeatureClass* ps)
{
	int i, j;
	int64_t tmp64;
	int16_t tmp;
	stftModule_setDefault(&ps->state_stftModule);
	for (i = 0; i < ps->dim_feat; i++)
	{
		tmp64 = (int64_t)((int32_t)LOG10_2POW_N15_Q15 - ps->pt_norm_mean[i]);
		tmp64 = (tmp64 * (int64_t)ps->pt_norm_stdR[i]) >> (30 - ps->qbit_output);
		tmp64 = MIN(MAX(tmp64, (int64_t)MIN_INT16_T), (int64_t)MAX_INT16_T);
		tmp = (int16_t) tmp64;

		for (j = 0; j < (ps->num_context - 1); j++)
		{
			ps->normFeatContext[i + j * ps->dim_feat] = tmp;
		}

	}
}

void FeatureClass_execute(
		FeatureClass*ps,
		int16_t* input)
{
	int32_t* pspec = GLOBAL_PSPEC;
	int32_t* spec = ps->state_stftModule.spec;
	int shift = (ps->num_context - 1) * ps->dim_feat;
	int i;
	int64_t tmp;
	for (i = 0; i < shift; i++)
	{
		ps->normFeatContext[i] = ps->normFeatContext[i + ps->dim_feat];
	}
#if ARM_FFT==0
	stftModule_analyze(&ps->state_stftModule, input, spec);
	spec2pspec(pspec, spec, 1 + (LEN_FFT_NNSP >> 1));
#else
	stftModule_analyze_arm(
		(void*) &ps->state_stftModule,
		input, // q15
		spec); // q21
	spec2pspec_arm(pspec, spec, 1 + (LEN_FFT_NNSP >> 1));
#endif
	melSpecProc(pspec, ps->feature, ps->p_melBanks, ps->num_mfltrBank);
	log10_vec(ps->feature, ps->feature, ps->dim_feat, 15);
	for (i = 0; i < ps->dim_feat; i++)
	{
		tmp = (int64_t) ps->feature[i] - (int64_t) ps->pt_norm_mean[i];
		tmp = (tmp * ((int64_t)ps->pt_norm_stdR[i])) >> (30 - ps->qbit_output); //Bit_frac_out = 30-22 = 8
		tmp = MIN(MAX(tmp, (int64_t) MIN_INT16_T), (int64_t) MAX_INT16_T);
		ps->normFeatContext[i + shift] = (int16_t) tmp;
	}

}