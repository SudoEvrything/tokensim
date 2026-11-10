#ifndef TOKENSIMD_SYS_H
#define TOKENSIMD_SYS_H

#include <stddef.h>
#include <stdint.h>

#include "batch.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tsimd_sys tsimd_sys;

tsimd_sys *tsimd_sys_open(const char *model_path, const char *format,
                          const char *scheme, char *err, size_t err_cap);
void tsimd_sys_close(tsimd_sys *h);

size_t tsimd_sys_vocab_size(const tsimd_sys *h);
const char *tsimd_sys_version(void);

int tsimd_sys_encode(tsimd_sys *h, const uint8_t *data, size_t len,
                     int threads, uint32_t **ids, size_t *n_ids);
int tsimd_sys_encode_into(tsimd_sys *h, const uint8_t *data, size_t len,
                          int threads, uint32_t *out, size_t cap,
                          size_t *n_out);
int tsimd_sys_encode_bound(tsimd_sys *h, const uint8_t *data, size_t len,
                           int threads, size_t *cap_out);
int tsimd_sys_decode_parallel(tsimd_sys *h, const uint32_t *ids, size_t n,
                              int skip_special, uint8_t *buf, size_t cap,
                              size_t *out_len, int threads);
int tsimd_sys_decode_into(tsimd_sys *h, const uint32_t *ids, size_t n,
                          int skip_special, uint8_t *buf, size_t cap,
                          size_t *out_len);
int tsimd_sys_decode(tsimd_sys *h, const uint32_t *ids, size_t n,
                     int skip_special, uint8_t **bytes, size_t *n_bytes);
int tsimd_sys_encode_batch(tsimd_sys *h, const tsimd_input_span_t *docs,
                           size_t n_docs, int threads, uint32_t **ids,
                           size_t *n_ids, size_t **offsets);
int tsimd_sys_encode_own(tsimd_sys *h, const uint8_t *data, size_t len,
                        int threads);
const uint32_t *tsimd_sys_ids(const tsimd_sys *h, size_t *n);

void tsimd_sys_free(void *p);

#ifdef __cplusplus
}
#endif

#endif
