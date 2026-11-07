#ifndef TSIMD_MODEL_INTERNAL_H
#define TSIMD_MODEL_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include "tokenizer.h"
#include "vocabulary.h"

int tsimd_decode_remapped(const char *s, size_t n, uint8_t *out, size_t cap,
                         uint32_t *out_len);

uint32_t tsimd_vocab_merge_target(const tsimd_vocab_t *v, uint32_t left,
                                 uint32_t right);

void tsimd_bpe_split_dump(void);

#endif

int tsimd_spm_parse(const char *path, char ***out_str, uint32_t **out_score,
                    int **out_type);

size_t tsimd_metaspace(const uint8_t *in, size_t len, uint8_t *out,
                       size_t cap);

size_t tsimd_metaspace_ex(const uint8_t *in, size_t len, uint8_t *out,
                          size_t cap, int dummy_prefix);

size_t tsimd_spm_normalize(const uint8_t *in, size_t len, uint8_t *out,
                           size_t cap, int dummy_prefix, int remove_extra);
