#include "bpe.h"
#include "tokenizer.h"
#include "unicode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static int checks = 0;

#define CK(cond, ...)                                  \
    do {                                               \
        checks++;                                      \
        if (!(cond)) {                                 \
            fprintf(stderr, "FAIL %d: ", __LINE__);    \
            fprintf(stderr, __VA_ARGS__);              \
            fprintf(stderr, "\n");                     \
            fails++;                                   \
        }                                              \
    } while (0)

static uint32_t sim_atoms(const tsimd_vocab_t *v, const uint8_t *b,
                          uint32_t len, uint32_t *parts) {
    uint32_t np = 0;
    for (uint32_t k = 0; k < len;) {
        uint32_t cp;
        tsimd_char_class_t cls;
        size_t a = tsimd_utf8_next(b + k, len - k, &cp, &cls);
        if (a == 0) a = 1;
        if (a == 1 || !v->raw_piece_units) {
            uint32_t pid = v->byte_ids[b[k]];
            if (pid == UINT32_MAX || pid >= v->vocab_size) return 0;
            parts[np++] = pid;
            k += 1;
            continue;
        }
        uint32_t pid = tsimd_vocab_find(v, b + k, (uint32_t)a);
        if (pid == UINT32_MAX || pid >= v->vocab_size) {
            for (uint32_t j = 0; j < (uint32_t)a; j++) {
                uint32_t bid = v->byte_ids[b[k + j]];
                if (bid == UINT32_MAX || bid >= v->vocab_size) return 0;
                parts[np++] = bid;
            }
            k += (uint32_t)a;
            continue;
        }
        parts[np++] = pid;
        k += (uint32_t)a;
    }
    return np;
}

static uint32_t sim_run(const tsimd_vocab_t *v, uint32_t *parts,
                        uint32_t np) {
    while (np > 1) {
        uint32_t best_r = UINT32_MAX;
        uint32_t bi = np;
        uint32_t mid = 0;
        for (uint32_t i = 0; i + 1 < np; i++) {
            uint32_t m = 0;
            uint32_t r = tsimd_vocab_merge_prio_id(v, parts[i], parts[i + 1],
                                                   &m);
            if (r < best_r) {
                best_r = r;
                bi = i;
                mid = m;
            }
        }
        if (best_r == UINT32_MAX) break;
        parts[bi] = mid;
        memmove(&parts[bi + 1], &parts[bi + 2],
                (np - bi - 2) * sizeof(uint32_t));
        np--;
    }
    return np;
}

static int sim_agrees(const tsimd_vocab_t *v, uint32_t id, uint32_t *parts,
                      uint32_t cap) {
    uint32_t len = v->offsets[id + 1] - v->offsets[id];
    if (len <= 1) return 1;
    if (len > cap) return 0;
    uint32_t np = sim_atoms(v, v->bytes + v->offsets[id], len, parts);
    if (np == 0) return 0;
    np = sim_run(v, parts, np);
    return np == 1 && parts[0] == id;
}

static void check_invariants(const tsimd_vocab_t *v, const char *name,
                             uint32_t stride, uint32_t *parts) {
    uint32_t max_len = 1;
    for (uint32_t id = 0; id < v->vocab_size; id++) {
        uint32_t l = v->offsets[id + 1] - v->offsets[id];
        if (l > max_len) max_len = l;
    }
    uint32_t bad_reach = 0, bad_sim = 0;
    for (uint32_t id = 0; id < v->vocab_size; id++) {
        uint32_t len = v->offsets[id + 1] - v->offsets[id];
        if (len < 2) continue;
        if (v->merge_sim[id] && v->merge_reach && !v->merge_reach[id])
            bad_reach++;
        if (id % stride == 0 && len <= max_len) {
            int want = sim_agrees(v, id, parts, max_len);
            if (want != (int)v->merge_sim[id]) bad_sim++;
        }
    }
    CK(bad_reach == 0, "%s: %u tokens sim-ok but not merge-reachable", name,
       bad_reach);
    CK(bad_sim == 0, "%s: %u sampled tokens disagree with the sim walk", name,
       bad_sim);
}

int main(void) {
    uint32_t *parts = malloc(65536 * sizeof(uint32_t));
    if (!parts) return 1;

    tsimd_tokenizer_t tok;
    memset(&tok, 0, sizeof tok);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(
            &tok, TSIMD_MODEL_TOKENIZER_JSON,
            TSIMD_SRC_DIR "/models/reference/gemma4_tokenizer.json",
            NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "FAIL: gemma4 load\n");
        return 1;
    }
    const tsimd_vocab_t *v = &tok.vocab;
    CK(v->merge_sim != NULL, "gemma4 merge_sim missing");
    if (v->merge_sim) {
        CK(v->merge_sim[160098] == 0,
           "gemma4 160098 ('^yyyy') must not be whole-piece (%u)",
           v->merge_sim[160098]);
        CK(v->merge_sim[570] == 1, "gemma4 570 must be whole-piece");
        CK(v->merge_sim[13828] == 1, "gemma4 13828 must be whole-piece");
        CK(v->merge_sim[236762] == 1, "gemma4 236762 must be whole-piece");
        CK(v->merge_sim[29104] == 1, "gemma4 29104 must be whole-piece");
    }
    {
        const uint8_t chunk[] = {0x20, 'y', 'y', 'y', 'y'};
        tsimd_token_sequence_t seq;
        memset(&seq, 0, sizeof seq);
        if (tsimd_tokenizer_encode(&tok, chunk, 5, &seq) != 0) {
            CK(0, "gemma4 ' yyyy' encode rc");
        } else {
            CK(seq.count == 3 && seq.ids[0] == 570 && seq.ids[1] == 13828 &&
                   seq.ids[2] == 236762,
               "gemma4 ' yyyy' encode: %zu ids:", seq.count);
            for (size_t i = 0; i < seq.count && i < 6; i++)
                fprintf(stderr, " %u", seq.ids[i]);
            fprintf(stderr, "\n");
            tsimd_token_sequence_free(&seq);
        }
    }
    check_invariants(v, "gemma4", 911, parts);
    tsimd_tokenizer_free(&tok);

    tsimd_tokenizer_t ds;
    memset(&ds, 0, sizeof ds);
    sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(
            &ds, TSIMD_MODEL_TOKENIZER_JSON,
            TSIMD_SRC_DIR "/models/reference/deepseek_tokenizer.json",
            NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "FAIL: deepseek load\n");
        return 1;
    }
    const tsimd_vocab_t *dv = &ds.vocab;
    CK(dv->merge_sim != NULL, "deepseek merge_sim missing");
    if (dv->merge_sim) {
        uint32_t the = tsimd_vocab_find(dv, (const uint8_t *)" the", 4);
        CK(the != UINT32_MAX && dv->merge_sim[the] == 1,
           "deepseek ' the' must be whole-piece (id %u)", the);
    }
    check_invariants(dv, "deepseek", 1031, parts);
    tsimd_tokenizer_free(&ds);

    free(parts);
    printf("test_wholepiece_sim: %d checks, %d fails\n", checks, fails);
    return fails != 0;
}
