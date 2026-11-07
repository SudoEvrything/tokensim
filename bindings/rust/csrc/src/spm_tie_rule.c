#include "spm_tie_rule.h"

#include <math.h>

#define SP_TIE_ULPS 8

float tsimd_spm_path_score(const tsimd_vocab_t *v, const uint32_t *path,
                           size_t n) {
    float sum = 0.0f;
    if (!v || !path || !v->scores) return 0.0f;
    for (size_t i = 0; i < n; i++) {
        if (path[i] >= v->vocab_size) return -INFINITY;
        sum += v->scores[path[i]];
    }
    return sum;
}

size_t tsimd_spm_path_unks(const tsimd_vocab_t *v, const uint32_t *path,
                          size_t n) {
    size_t k = 0;
    if (!v || !path) return (size_t)-1;
    for (size_t i = 0; i < n; i++)
        if (path[i] == v->spm_unk_id) k++;
    return k;
}

int tsimd_spm_accepts_reference_score(const tsimd_vocab_t *v,
                                      const uint32_t *mine, size_t mine_n,
                                      const uint32_t *ref, size_t ref_n,
                                      float mine_sum, float ref_sum) {
    size_t mu, ru;
    float mag, tol, diff;
    if (!v || !mine || !ref) return 0;
    if (mine_n != ref_n) return 0;
    if (isnan(mine_sum) || isnan(ref_sum)) return 0;
    mu = tsimd_spm_path_unks(v, mine, mine_n);
    ru = tsimd_spm_path_unks(v, ref, ref_n);
    if (mu == (size_t)-1 || ru == (size_t)-1) return 0;
    if (mu != ru) return 0;
    mag = fabsf(mine_sum) + fabsf(ref_sum);
    tol = mag > 0.0f ? (float)SP_TIE_ULPS * mag * 5.9604645e-8f : 1e-30f;
    diff = fabsf(mine_sum - ref_sum);
    return diff <= tol;
}
