#ifndef TSIMD_TOKENIZER_H
#define TSIMD_TOKENIZER_H

#include <stddef.h>
#include <stdatomic.h>
#include <stdint.h>
#include "vocabulary.h"
#include "pretokenizer.h"
#include "bpe.h"
#include "batch.h"

typedef enum {
    TSIMD_MODEL_TIKTOKEN = 0,
    TSIMD_MODEL_GPT2,

    TSIMD_MODEL_TOKENIZER_JSON,

    TSIMD_MODEL_SENTENCEPIECE

} tsimd_model_format_t;

typedef struct {
    const char *text;
    uint32_t id;
    uint32_t length;
    int is_special;
} tsimd_added_token_t;

typedef struct {
    size_t off;
    uint32_t len;
    uint32_t idx;
} tsimd_added_seam_t;

typedef struct {
    tsimd_added_seam_t inline_items[32];
    tsimd_added_seam_t *items;
    size_t count;
    size_t cap;
} tsimd_added_seam_list_t;

typedef struct tsimd_tokenizer tsimd_tokenizer_t;

int tsimd_tokenizer_added_seams(const tsimd_tokenizer_t *t,
                                const uint8_t *data, size_t len,
                                tsimd_added_seam_list_t *out);
void tsimd_added_seam_list_free(tsimd_added_seam_list_t *list);

typedef struct tsimd_pool tsimd_pool_t;

#define TSIMD_TOKENIZER_MAGIC 0x5453494Du

#define TSIMD_NORM_NONE 0
#define TSIMD_NORM_NFC  1
#define TSIMD_NORM_NFKC 2

typedef struct tsimd_tokenizer {
    uint32_t magic_;
    tsimd_vocab_t vocab;
    tsimd_pretok_scheme_t scheme;

    int spm_kind;
    int spm_dummy_prefix;
    int spm_remove_extra;

    int norm;
    tsimd_added_token_t *added;
    size_t added_count;
    tsimd_added_token_t **added_by_id;
    size_t added_by_id_n;
    uint8_t *spm_ctl_flags;
    size_t spm_ctl_n;
    size_t cache_bytes;

    uint16_t added_head[256];
    uint16_t *added_head2;
    uint16_t *added_next;
    int added_owned;
    char *added_text_blob;

    tsimd_pretok_cache_t cache;

    tsimd_pretok_cache_t *spm_caches;
    size_t spm_caches_n;

    tsimd_pretok_cache_t memo_cache;

    tsimd_pool_t *pool;

    void *_Atomic seg_state;
    _Atomic int seg_busy;
} tsimd_tokenizer_t;

int tsimd_tokenizer_load_model(tsimd_tokenizer_t *t, tsimd_model_format_t fmt,
                              const char *path1, const char *path2,
                              tsimd_pretok_scheme_t *scheme,
                              const tsimd_added_token_t *extra,
                              size_t n_extra);

int tsimd_tokenizer_load(tsimd_tokenizer_t *t, const char *vocab_path,
                        tsimd_pretok_scheme_t scheme,
                        const tsimd_added_token_t *specials,
                        size_t n_specials);

void tsimd_tokenizer_free(tsimd_tokenizer_t *t);

size_t tsimd_tokenizer_next_added(const tsimd_tokenizer_t *t,
                                 const uint8_t *data, size_t len,
                                 size_t pos, size_t *added_index);

size_t tsimd_tokenizer_added_len_at(const tsimd_tokenizer_t *t,
                                    const uint8_t *data, size_t len,
                                    size_t pos, size_t *added_index);

int tsimd_encode_segment_pool(tsimd_tokenizer_t *t, const uint8_t *data,
                              size_t len, tsimd_bpe_workspace_t *ws,
                              tsimd_pretok_cache_t *c,
                              tsimd_pretok_cache_t *shared);

int tsimd_encode_buffer_pool(tsimd_tokenizer_t *t, const uint8_t *data,
                             size_t len, tsimd_bpe_workspace_t *ws,
                             tsimd_pretok_cache_t *c,
                             tsimd_pretok_cache_t *shared);

typedef struct {
    uint32_t *ids;
    size_t count;
    size_t cap;
} tsimd_token_sequence_t;

int tsimd_tokenizer_encode(tsimd_tokenizer_t *t, const uint8_t *data,
                          size_t len, tsimd_token_sequence_t *out);

int tsimd_tokenizer_encode_into(tsimd_tokenizer_t *t, const uint8_t *data,
                               size_t len, tsimd_token_sequence_t *seq);

size_t tsimd_tokenizer_encode_bound(const tsimd_tokenizer_t *t, size_t len);

int tsimd_tokenizer_encode_into_raw(tsimd_tokenizer_t *t, const uint8_t *data,
                                   size_t len, uint32_t *buf, size_t cap,
                                   size_t *count);

void tsimd_token_sequence_free(tsimd_token_sequence_t *seq);

typedef struct {
    uint8_t *bytes;
    size_t len;
} tsimd_byte_sequence_t;

int tsimd_tokenizer_decode(const tsimd_tokenizer_t *t,
                           const uint32_t *ids, size_t n, int skip_special,
                           tsimd_byte_sequence_t *out);
void tsimd_byte_sequence_free(tsimd_byte_sequence_t *seq);

int tsimd_tokenizer_encode_parallel_into(tsimd_tokenizer_t *t,
                                        const uint8_t *data, size_t len,
                                        int max_threads, uint32_t *buf,
                                        size_t cap, size_t *count);

int tsimd_tokenizer_encode_parallel(tsimd_tokenizer_t *t,
                                   const uint8_t *data, size_t len,
                                   int max_threads,
                                   tsimd_token_sequence_t *out);

int tsimd_default_threads(void);

int tsimd_tokenizer_encode_whole_chain_par(tsimd_tokenizer_t *t,
                                          const uint8_t *data, size_t len,
                                          int threads,
                                          tsimd_token_sequence_t *out);

uint8_t *tsimd_nfc_normalize_all(const tsimd_tokenizer_t *t,
                                 const uint8_t *data, size_t len,
                                 int max_threads, size_t *out_len);

uint8_t *tsimd_nfc_normalize_all_tl(const tsimd_tokenizer_t *t,
                                    const uint8_t *data, size_t len,
                                    int max_threads, size_t *out_len);

int tsimd_tokenizer_encode_batch(tsimd_tokenizer_t *t,
                                const tsimd_input_span_t *docs,
                                size_t n_docs, tsimd_token_batch_t *out,
                                int max_threads);
void tsimd_token_batch_free(tsimd_token_batch_t *batch);

void tsimd_tokenizer_release_pool(tsimd_tokenizer_t *t);

void tsimd_restore_process_affinity(void);

void tsimd_pin_thread_nth(int idx);

void tsimd_nfc_tls_release(void);
int tsimd_encode_chunk_nfc_fused(tsimd_tokenizer_t *t,
                                 tsimd_bpe_workspace_t *ws,
                                 const uint8_t *p, size_t n, int compat);

size_t tsimd_nfc_seam_at(const tsimd_tokenizer_t *t, const uint8_t *data,
                             size_t len, size_t from);

tsimd_pretok_cache_t *tsimd_spm_worker_cache(tsimd_tokenizer_t *t, size_t idx,
                                            size_t chunk_len);

void tsimd_tokenizer_cache_stats(const tsimd_tokenizer_t *t,
                                 size_t *hits, size_t *misses,
                                 size_t *inserts);

int tsimd_par_split(const size_t *weights, size_t n, int width,
                    size_t *lo, size_t *hi);
int tsimd_par_drain_probe(size_t n, int width, const uint32_t *spin,
                          unsigned *runs);

#endif
