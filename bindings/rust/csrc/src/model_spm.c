#include "model_internal.h"
#include "tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SPM_PIECE_MAX 1024

typedef struct {
    const char *str;
    uint32_t len;
    float score;
    int type;
} spm_piece_t;

static size_t spm_varint(const uint8_t *b, size_t i, size_t end,
                         uint64_t *out) {
    uint64_t v = 0;
    unsigned s = 0;
    while (i < end) {
        uint8_t x = b[i++];
        v |= (uint64_t)(x & 0x7F) << s;
        if (x < 0x80) { *out = v; return i; }
        s += 7;
    }
    return 0;
}

static int spm_parse_piece(const uint8_t *b, size_t i, size_t end,
                           spm_piece_t *out) {
    const uint8_t *str = NULL;
    uint32_t slen = 0;
    float score = 0.0f;
    int type = 0;
    while (i < end) {
        uint64_t tag;
        size_t ni = spm_varint(b, i, end, &tag);
        if (!ni) return -1;
        i = ni;
        unsigned f = (unsigned)(tag >> 3), wt = (unsigned)(tag & 7);
        if (wt == 2) {
            uint64_t n;
            ni = spm_varint(b, i, end, &n);
            if (!ni || n > SPM_PIECE_MAX) return -1;
            i = ni;
            if (f == 1) { str = (const char *)b + i; slen = (uint32_t)n; }
            i += n;
        } else if (wt == 5 && f == 2) {
            uint32_t raw;
            if (i + 4 > end) return -1;
            memcpy(&raw, b + i, 4);
            score = *((const float *)&raw);
            i += 4;
        } else if (wt == 0 && f == 3) {
            uint64_t v;
            ni = spm_varint(b, i, end, &v);
            if (!ni) return -1;
            i = ni;
            type = (int)v;
        } else {
            return -1;
        }
    }
    out->str = str;
    out->len = slen;
    out->score = score;
    out->type = type;
    return 0;
}

int tsimd_spm_parse(const char *path, char ***out_str, uint32_t **out_score,
                    int **out_type) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long flen = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = malloc((size_t)flen);
    if (!data) { fclose(f); return -1; }
    size_t got = fread(data, 1, (size_t)flen, f);
    fclose(f);
    if (got != (size_t)flen) { free(data); return -1; }

    size_t cap = 4096, n = 0;
    char **strs = malloc(cap * sizeof(char *));
    uint32_t *scores = malloc(cap * sizeof(uint32_t));
    int *types = malloc(cap * sizeof(int));
    if (!strs || !scores || !types) {
        free(data);
        free(strs);
        free(scores);
        free(types);
        return -1;
    }

    size_t i = 0;
    while (i < got) {
        uint64_t tag;
        size_t ni = spm_varint(data, i, got, &tag);
        if (!ni) break;
        i = ni;
        unsigned f = (unsigned)(tag >> 3), wt = (unsigned)(tag & 7);
        if (f != 1 || wt != 2) continue;
        uint64_t plen;
        ni = spm_varint(data, i, got, &plen);
        if (!ni) break;
        i = ni;
        spm_piece_t p;
        if (spm_parse_piece(data, i, i + plen, &p) != 0) break;
        i += plen;
        if (n == cap) {
            cap *= 2;
            char **ns = realloc(strs, cap * sizeof(char *));
            if (!ns) goto oom;
            strs = ns;
            uint32_t *ns2 = realloc(scores, cap * sizeof(uint32_t));
            if (!ns2) goto oom;
            scores = ns2;
            int *nt = realloc(types, cap * sizeof(int));
            if (!nt) goto oom;
            types = nt;
        }
        strs[n] = malloc((size_t)p.len + 1);
        if (!strs[n]) goto oom;
        memcpy(strs[n], p.str, p.len);
        strs[n][p.len] = 0;
        memcpy(&scores[n], &p.score, 4);
        types[n] = p.type;
        n++;
    }
    free(data);
    *out_str = strs;
    *out_score = scores;
    *out_type = types;
    return (int)n;
oom:
    for (size_t k = 0; k < n; k++) free(strs[k]);
    free(strs);
    free(scores);
    free(types);
    free(data);
    return -1;
}

static uint8_t spm_hexval(uint8_t c) {
    if (c >= '0' && c <= '9') return (uint8_t)(c - '0');
    if (c >= 'a' && c <= 'f') return (uint8_t)(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return (uint8_t)(c - 'A' + 10);
    return 0;
}

static int spm_byte_fallback(const char *s, size_t n, uint8_t *out) {
    if (n != 6 || s[0] != '<' || s[1] != '0' || s[2] != 'x' ||
        s[5] != '>')
        return 0;
    *out = (uint8_t)((spm_hexval((uint8_t)s[3]) << 4) |
                     spm_hexval((uint8_t)s[4]));
    return 1;
}

typedef struct {
    int model_type;
    int add_dummy_prefix;
    int remove_extra_ws;
    int unk_id;
    char unk_surface[16];
    unsigned unk_surface_len;
} spm_spec_t;

static uint64_t spm_varint_u(const uint8_t *b, size_t i, size_t end,
                             unsigned *adv) {
    uint64_t v = 0;
    unsigned s = 0;
    size_t j = i;
    while (j < end && s < 64) {
        uint8_t x = b[j++];
        v |= (uint64_t)(x & 0x7F) << s;
        if (x < 0x80) { *adv = (unsigned)(j - i); return v; }
        s += 7;
    }
    *adv = 0;
    return 0;
}

static int spm_spec_parse(const uint8_t *data, size_t got, spm_spec_t *out) {
    out->model_type = 2;
    out->add_dummy_prefix = 1;
    out->remove_extra_ws = 0;
    out->unk_id = 0;
    memcpy(out->unk_surface, " \xE2\x81\x87 ", 5);
    out->unk_surface_len = 5;
    size_t i = 0;
    while (i < got) {
        unsigned adv = 0;
        uint64_t tag = spm_varint_u(data, i, got, &adv);
        i += adv;
        if (adv == 0) break;
        unsigned f = (unsigned)(tag >> 3), wt = (unsigned)(tag & 7);
        if (wt != 2) {
            if (wt == 0) {
                unsigned a2 = 0;
                spm_varint_u(data, i, got, &a2);
                i += a2;
            } else if (wt == 5) {
                i += 4;
            } else {
                break;
            }
            continue;
        }
        uint64_t n;
        unsigned a3 = 0;
        n = spm_varint_u(data, i, got, &a3);
        i += a3;
        if (a3 == 0 || i + n > got) break;
        const uint8_t *sub = data + i;
        i += (size_t)n;
        if (f == 2) {
            size_t j = 0;
            while (j < n) {
                unsigned av = 0;
                uint64_t st = spm_varint_u(sub, j, n, &av);
                j += av;
                if (av == 0) break;
                unsigned sf = (unsigned)(st >> 3), swt = (unsigned)(st & 7);
                if (swt == 0) {
                    uint64_t vv;
                    unsigned a4 = 0;
                    vv = spm_varint_u(sub, j, n, &a4);
                    j += a4;
                    if (sf == 3 && a4 > 0) out->model_type = (int)vv;
                    if (sf == 40 && a4 > 0) out->unk_id = (int)vv;
                } else if (swt == 2) {
                    unsigned a5 = 0;
                    uint64_t ln = spm_varint_u(sub, j, n, &a5);
                    if (sf == 44 && ln > 0 && ln <= 15 && a5 > 0 &&
                        j + a5 + (size_t)ln <= n) {
                        memcpy(out->unk_surface, sub + j + a5, (size_t)ln);
                        out->unk_surface_len = (unsigned)ln;
                    }
                    j += a5 + (size_t)ln;
                } else if (swt == 5) {
                    j += 4;
                } else {
                    break;
                }
            }
        } else if (f == 3) {
            size_t j = 0;
            while (j < n) {
                unsigned av = 0;
                uint64_t st = spm_varint_u(sub, j, n, &av);
                j += av;
                if (av == 0) break;
                unsigned sf = (unsigned)(st >> 3), swt = (unsigned)(st & 7);
                if (swt == 0) {
                    uint64_t vv;
                    unsigned a4 = 0;
                    vv = spm_varint_u(sub, j, n, &a4);
                    j += a4;
                    if (sf == 3) out->add_dummy_prefix = vv ? 1 : 0;
                    if (sf == 4) out->remove_extra_ws = vv ? 1 : 0;
                } else if (swt == 2) {
                    unsigned a5 = 0;
                    uint64_t ln = spm_varint_u(sub, j, n, &a5);
                    j += a5 + (size_t)ln;
                } else if (swt == 5) {
                    j += 4;
                } else {
                    break;
                }
            }
        }
    }
    return 0;
}

typedef struct {
    uint32_t isnorm;
    float score;
    int idx;
} spm_rank_key_t;

static int spm_rank_cmp(const void *A, const void *B) {
    const spm_rank_key_t *x = (const spm_rank_key_t *)A;
    const spm_rank_key_t *y = (const spm_rank_key_t *)B;
    if (x->isnorm != y->isnorm) return y->isnorm - x->isnorm;
    if (x->isnorm && x->score != y->score) return x->score < y->score ? -1 : 1;
    if (x->idx != y->idx) return x->idx < y->idx ? -1 : 1;
    return 0;
}

int tsimd_model_load_spm(tsimd_tokenizer_t *t, const char *path,
                         const char *fp) {
    (void)fp;
    char **strs = NULL;
    uint32_t *scores = NULL;
    int *types = NULL;
    int *score_order = NULL;
    uint32_t *rank = NULL;
    spm_spec_t spec;
    int n = tsimd_spm_parse(path, &strs, &scores, &types);
    int rc = -1;
    if (n < 0) return -1;

    {
        FILE *f = fopen(path, "rb");
        uint8_t *raw = NULL;
        long flen = 0;
        if (f) {
            fseek(f, 0, SEEK_END);
            flen = ftell(f);
            fseek(f, 0, SEEK_SET);
            raw = malloc((size_t)flen);
            if (raw && fread(raw, 1, (size_t)flen, f) != (size_t)flen) {
                free(raw);
                raw = NULL;
            }
            fclose(f);
        }
        memset(&spec, 0, sizeof spec);
        if (raw) {
            spm_spec_parse(raw, (size_t)flen, &spec);
            free(raw);
        }
    }
    t->spm_dummy_prefix = spec.add_dummy_prefix;
    t->spm_remove_extra = spec.remove_extra_ws;
    t->spm_kind = spec.model_type == 1;
    if (tsimd_vocab_build_start(&t->vocab) != 0) goto fail;

    score_order = malloc((size_t)n * sizeof(int));
    if (!score_order) goto fail;
    for (int i = 0; i < n; i++) score_order[i] = i;

    {
        spm_rank_key_t *keys = malloc((size_t)n * sizeof(*keys));
        if (!keys) {
            free(score_order);
            goto fail;
        }
        for (int i = 0; i < n; i++) {
            keys[i].isnorm = types[i] == 0;
            keys[i].score = (float)scores[i];
            keys[i].idx = i;
        }
        qsort(keys, (size_t)n, sizeof(*keys), spm_rank_cmp);
        for (int i = 0; i < n; i++) score_order[i] = keys[i].idx;
        free(keys);
    }

    rank = malloc((size_t)n * sizeof(uint32_t));
    if (!rank) goto fail;
    {
        uint32_t r = 0;
        for (int i = 0; i < n; i++) {
            if (i > 0 &&
                scores[score_order[i]] != scores[score_order[i - 1]])
                r++;
            rank[score_order[i]] = r;
        }
    }
    for (int i = 0; i < n; i++) {
        size_t sl = strlen(strs[i]);
        uint8_t b;

        if (spm_byte_fallback(strs[i], sl, &b)) {
            if (tsimd_vocab_add_token(&t->vocab, &b, 1, (uint32_t)i) ==
                UINT32_MAX)
                goto fail;
        } else if (types[i] == 0) {

            if (tsimd_vocab_add_token(&t->vocab, (const uint8_t *)strs[i],
                                      (uint32_t)sl, (uint32_t)i) ==
                UINT32_MAX)
                goto fail;
        }
    }

    if (tsimd_vocab_seal_ex(&t->vocab, 0, t->spm_kind ? 0 : 1) != 0) goto fail;
    t->vocab.spm_unk_id = (uint32_t)spec.unk_id;
    if (spec.unk_surface_len == 0) {
        memcpy(spec.unk_surface, " \xE2\x81\x87 ", 5);
        spec.unk_surface_len = 5;
    }
    memcpy(t->vocab.spm_unk_surface, spec.unk_surface, spec.unk_surface_len);
    t->vocab.spm_unk_surface_len = (uint8_t)spec.unk_surface_len;
    for (int i = 0; i < n; i++) {
        if (types[i] == 3 && t->vocab.spm_control_n < 8)
            t->vocab.spm_control_ids[t->vocab.spm_control_n++] = (uint32_t)i;
    }

    if (t->spm_kind) {
        float *sc = calloc((size_t)n, sizeof(float));
        if (!sc) goto fail;
        t->vocab.scores = sc;
        float mn = INFINITY;
        for (int i = 0; i < n; i++) {
            float f;
            memcpy(&f, &scores[i], 4);
            sc[i] = f;
            if (types[i] == 0 && f < mn) mn = f;
        }
        t->vocab.spm_min_score = mn;
        tsimd_vocab_build_spm_trie(&t->vocab);
        rc = 0;
        goto fail;
    }

    for (uint32_t i = 0; i < (uint32_t)n; i++)
        if (types[i] == 0 || types[i] == 6)
            tsimd_vocab_set_rank(&t->vocab, i, rank[i]);

    for (uint32_t i = 0; i < (uint32_t)n; i++) {
        if (types[i] != 0) continue;
        size_t sl = strlen(strs[i]);
        if (sl == 1)
            t->vocab.byte_ids[(uint8_t)strs[i][0]] = i;
    }

    for (uint32_t ri = t->vocab.vocab_size; ri-- > 0;) {
        uint32_t id = ri;
        uint32_t plen = t->vocab.offsets[id + 1] - t->vocab.offsets[id];
        if (plen < 2) continue;
        const uint8_t *pb = t->vocab.bytes + t->vocab.offsets[id];

        for (uint32_t we = 1; we < plen; we++) {
            size_t rl = plen - we;
            uint32_t left = we == 1
                                ? t->vocab.byte_ids[pb[0]]
                                : tsimd_vocab_find(&t->vocab, pb, we);
            uint32_t right = rl == 1
                                 ? t->vocab.byte_ids[pb[we]]
                                 : tsimd_vocab_find(&t->vocab, pb + we,
                                                    (uint32_t)rl);
            if (left == UINT32_MAX || right == UINT32_MAX) continue;
            if (tsimd_vocab_record_merge(&t->vocab, left, right, id) != 0)
                goto fail;
        }
    }
    if (tsimd_vocab_pair_rank_build_slots(&t->vocab) != 0) goto fail;
    if (t->spm_kind) {
        (void)tsimd_vocab_spm_cross_ready(&t->vocab);
        (void)tsimd_vocab_spm_seam_ok(&t->vocab);
    }
    rc = 0;
fail:
    for (int i = 0; i < n; i++) free(strs[i]);

    free(strs);
    free(scores);
    free(types);
    free(score_order);
    free(rank);
    return rc;
}
