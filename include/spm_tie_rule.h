#ifndef TSIMD_SPM_TIE_RULE_H
#define TSIMD_SPM_TIE_RULE_H

#include "vocabulary.h"

#include <stddef.h>
#include <stdint.h>

float tsimd_spm_path_score(const tsimd_vocab_t *v, const uint32_t *path,
                           size_t n);

size_t tsimd_spm_path_unks(const tsimd_vocab_t *v, const uint32_t *path,
                           size_t n);

int tsimd_spm_accepts_reference_score(const tsimd_vocab_t *v,
                                     const uint32_t *mine, size_t mine_n,
                                     const uint32_t *ref, size_t ref_n,
                                     float mine_sum, float ref_sum);

#endif
