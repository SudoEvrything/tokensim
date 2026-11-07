#include "tokenizer.h"
#include "vocabulary.h"
#include "input.h"
#include "allocator.h"
#include "json.h"
#include "byte_unicode.h"
#include "model_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int add_json_token(tsimd_vocab_t *v, const char *s, uint32_t id,
                          uint8_t **scratch, size_t *scratch_cap) {
    size_t n = strlen(s);
    if (n + 1 > *scratch_cap) {
        *scratch_cap = n + 64;
        *scratch = tsimd_xrealloc(*scratch, *scratch_cap);
    }
    uint32_t blen;
    if (tsimd_decode_remapped(s, n, *scratch, *scratch_cap, &blen) != 0)
        return -1;
    return tsimd_vocab_add_token(v, *scratch, blen, id) == UINT32_MAX ? -1 : 0;
}

static int load_encoder_json(tsimd_vocab_t *v, const char *path) {
    tsimd_input_t in;
    if (tsimd_input_read_file(path, &in) != 0) {
        fprintf(stderr, "tsimd: cannot read %s\n", path);
        return -1;
    }
    tsimd_arena_t arena;
    tsimd_arena_init(&arena);
    size_t err_off;
    const tsimd_json_value_t *root =
        tsimd_json_parse(&arena, (const char *)in.data, in.length, &err_off);
    int rc = -1;
    uint8_t *scratch = NULL;
    size_t scratch_cap = 0;
    if (!root || root->type != TSIMD_JSON_OBJECT) {
        fprintf(stderr, "tsimd: %s:%zu invalid JSON vocabulary\n", path, err_off);
        goto out;
    }
    for (size_t i = 0; i < root->u.obj.n; i++) {
        const char *key = root->u.obj.keys[i];
        const tsimd_json_value_t *val = root->u.obj.vals[i];
        if (val->type != TSIMD_JSON_INT) goto out;
        long long id = val->u.i;
        if (id < 0 || id > 0xFFFFFFFFLL) goto out;
        if (add_json_token(v, key, (uint32_t)id, &scratch, &scratch_cap) != 0) {
            fprintf(stderr, "tsimd: invalid token string or duplicate id in %s\n",
                    path);
            goto out;
        }
    }
    rc = 0;
out:
    free(scratch);
    tsimd_arena_reset(&arena);
    tsimd_input_release(&in);
    return rc;
}

static size_t count_merge_lines(const tsimd_input_t *in) {
    size_t count = 0;
    const char *p = (const char *)in->data;
    const char *end = p + in->length;
    int first = 1;
    while (p < end) {
        const char *nl = memchr(p, '\n', (size_t)(end - p));
        if (!nl) break;
        if (!(first && nl - p >= 1 && p[0] == '#')) count++;
        first = 0;
        p = nl + 1;
    }
    return count;
}

static int record_merges(tsimd_vocab_t *v, const char *path, const char *data,
                         size_t len) {
    const char *p = data, *end = data + len;
    uint32_t rank = 0;
    int first = 1;
    while (p < end) {
        const char *nl = memchr(p, '\n', (size_t)(end - p));
        if (!nl) break;
        size_t line = (size_t)(nl - p);
        if (line > 0 && nl[-1] == '\r') line--;
        if (first && line >= 1 && p[0] == '#') {
            first = 0;
            p = nl + 1;
            continue;
        }
        first = 0;
        if (line > 0) {
            const char *sp = memchr(p, ' ', line);
            if (!sp) {
                fprintf(stderr, "tsimd: %s:%u malformed merge line\n", path,
                        rank + 1);
                return -1;
            }
            uint8_t lb[1024], rb[1024];
            uint32_t llen, rlen;
            if (tsimd_decode_remapped(p, (size_t)(sp - p), lb, sizeof(lb),
                                     &llen) != 0 ||
                tsimd_decode_remapped(sp + 1, (size_t)(p + line - sp - 1), rb,
                                     sizeof(rb), &rlen) != 0) {
                fprintf(stderr, "tsimd: %s:%u merge side outside byte alphabet\n",
                        path, rank + 1);
                return -1;
            }
            uint32_t l = tsimd_vocab_find(v, lb, llen);
            uint32_t r = tsimd_vocab_find(v, rb, rlen);
            if (l == UINT32_MAX || r == UINT32_MAX) {
                fprintf(stderr, "tsimd: %s:%u merge side not in vocabulary\n",
                        path, rank + 1);
                return -1;
            }
            uint32_t m = tsimd_vocab_merge_target(v, l, r);
            if (m == UINT32_MAX) {
                fprintf(stderr, "tsimd: %s:%u merge target not in vocabulary\n",
                        path, rank + 1);
                return -1;
            }
            if (tsimd_vocab_record_merge(v, l, r, m) != 0) return -1;
            tsimd_vocab_set_rank(v, m, rank);
            rank++;
        }
        p = nl + 1;
    }
    return 0;
}

static int load_merges(tsimd_vocab_t *v, const char *path) {
    tsimd_input_t in;
    if (tsimd_input_read_file(path, &in) != 0) {
        fprintf(stderr, "tsimd: cannot read %s\n", path);
        return -1;
    }
    int rc = -1;
    size_t n_merges = count_merge_lines(&in);
    if (n_merges == 0) {
        fprintf(stderr, "tsimd: %s contains no merges\n", path);
        goto out;
    }
    tsimd_vocab_mark_bhash_heavy(v);
    if (tsimd_vocab_seal(v, n_merges) != 0) {
        fprintf(stderr, "tsimd: %s vocabulary is not byte-complete\n", path);
        goto out;
    }
    rc = record_merges(v, path, (const char *)in.data, in.length);
    if (rc == 0 && tsimd_vocab_pair_rank_build(v) != 0) rc = -1;
    if (rc == 0 && tsimd_vocab_sim_build(v) != 0) rc = -1;
out:
    tsimd_input_release(&in);
    return rc;
}

static char *derive_merges_path(const char *encoder_path) {
    const char *slash = strrchr(encoder_path, '/');
    size_t dir_len = slash ? (size_t)(slash - encoder_path) + 1 : 0;
    static const char *names[] = {"vocab.bpe", "merges.txt"};
    for (size_t i = 0; i < 2; i++) {
        size_t nlen = strlen(names[i]);
        char *path = tsimd_xmalloc(dir_len + nlen + 1);
        memcpy(path, encoder_path, dir_len);
        memcpy(path + dir_len, names[i], nlen + 1);
        FILE *f = fopen(path, "rb");
        if (f) {
            fclose(f);
            return path;
        }
        free(path);
    }
    return NULL;
}

int tsimd_model_load_gpt2(tsimd_vocab_t *v, const char *encoder_path,
                         const char *merges_path_in) {
    tsimd_byteunicode_init();
    if (tsimd_vocab_build_start(v) != 0) return -1;

    char *merges_derived = NULL;
    const char *merges_path = merges_path_in;
    if (!merges_path) {
        merges_derived = derive_merges_path(encoder_path);
        if (!merges_derived) {
            fprintf(stderr,
                    "tsimd: no vocab.bpe/merges.txt next to %s (pass it explicitly)\n",
                    encoder_path);
            return -1;
        }
        merges_path = merges_derived;
    }

    int rc = load_encoder_json(v, encoder_path);
    if (rc == 0) rc = load_merges(v, merges_path);
    if (rc != 0) tsimd_vocab_free(v);
    free(merges_derived);
    return rc;
}
