#include "tokenizer.h"
#include "bpe.h"
#include "vocabulary.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GEMMA_MODEL TSIMD_SRC_DIR "/models/reference/gemma4_tokenizer.json"
#define MISTRAL_MODEL TSIMD_SRC_DIR "/models/reference/mistral7b_tokenizer.model"
#define QWEN_MODEL TSIMD_SRC_DIR "/models/reference/qwen2.5-0.5b_tokenizer.json"
#define SMOLLM_MODEL TSIMD_SRC_DIR "/models/reference/smollm2_tokenizer.json"

static int g_fail;
static int checks;

static void chk(int cond, const char *what, unsigned long a, unsigned long b) {
    checks++;
    if (!cond) {
        fprintf(stderr, "FAIL %s: %lu vs %lu\n", what, a, b);
        g_fail = 1;
    }
}

static uint64_t rng_state = 0x243F6A8885A308D3ull;
static uint64_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static int build_synth(tsimd_vocab_t *v) {
    memset(v, 0, sizeof *v);
    if (tsimd_vocab_build_start(v) != 0) return -1;
    for (int b = 0; b < 256; b++) {
        uint8_t c = (uint8_t)b;
        if (tsimd_vocab_add_token(v, &c, 1, (uint32_t)b) == UINT32_MAX)
            return -1;
    }
    if (tsimd_vocab_add_token(v, (const uint8_t *)"ab", 2, 256) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"xy", 2, 257) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"cd", 2, 258) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"abab", 4, 259) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_seal(v, 4) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'a', 'b', 256) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'x', 'y', 257) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'c', 'd', 258) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 256, 256, 259) != 0) return -1;
    tsimd_vocab_set_rank(v, 256, 0);
    tsimd_vocab_set_rank(v, 257, 0);
    tsimd_vocab_set_rank(v, 258, 1);
    tsimd_vocab_set_rank(v, 259, 2);
    if (tsimd_vocab_pair_rank_build(v) != 0) return -1;
    return 0;
}

static void check_synth(void) {
    tsimd_vocab_t v;
    if (build_synth(&v) != 0) {
        fprintf(stderr, "FAIL: synthetic vocab build\n");
        g_fail = 1;
        return;
    }
    uint32_t id;
    uint32_t p;

    id = 0;
    p = tsimd_vocab_merge_prio_id(&v, 'a', 'b', &id);
    chk(p == 0 && id == 256, "synth ab (prio,id)", (p << 16) | id, 256);
    id = 0;
    p = tsimd_vocab_merge_prio_id(&v, 'x', 'y', &id);
    chk(p == 0 && id == 257, "synth xy (prio,id)", (p << 16) | id, 257);

    chk(tsimd_vocab_id_of_priority(&v, 0) == 256 ||
            tsimd_vocab_id_of_priority(&v, 0) == 257,
        "synth inverse is ambiguous", tsimd_vocab_id_of_priority(&v, 0),
        tsimd_vocab_id_of_priority(&v, 0));

    id = 0;
    p = tsimd_vocab_merge_prio_id(&v, 'c', 'd', &id);
    chk(p == 1 && id == 258, "synth cd (prio,id)", (p << 16) | id, 1 << 16 | 258);

    id = 0;
    p = tsimd_vocab_merge_prio_id(&v, 256, 256, &id);
    chk(p == 2 && id == 259, "synth abab (prio,id)", (p << 16) | id, 2 << 16 | 259);

    id = 12345;
    p = tsimd_vocab_merge_prio_id(&v, 'a', 'c', &id);
    chk(p == UINT32_MAX, "synth ac is a miss", p, UINT32_MAX);
    chk(tsimd_vocab_merge_id(&v, 'a', 'c') == UINT32_MAX,
        "synth ac oracle miss", tsimd_vocab_merge_id(&v, 'a', 'c'), 0);
    chk(tsimd_vocab_merge_priority(&v, 'a', 'c') == UINT32_MAX,
        "synth ac priority miss", tsimd_vocab_merge_priority(&v, 'a', 'c'), 0);

    {
        static const uint32_t lp[4] = {'a', 'x', 'c', 256};
        static const uint32_t rp[4] = {'b', 'y', 'd', 256};
        for (int i = 0; i < 4; i++) {
            uint32_t pid = 0;
            uint32_t pp = tsimd_vocab_merge_prio_id(&v, lp[i], rp[i], &pid);
            chk(pp == tsimd_vocab_merge_priority(&v, lp[i], rp[i]),
                "synth priority == oracle", pp,
                tsimd_vocab_merge_priority(&v, lp[i], rp[i]));
            chk(pid == tsimd_vocab_merge_id(&v, lp[i], rp[i]),
                "synth id == oracle", pid,
                tsimd_vocab_merge_id(&v, lp[i], rp[i]));
        }
    }
    tsimd_vocab_free(&v);
}

static size_t count_ties(const tsimd_vocab_t *v, size_t *out_of_range) {
    size_t ties = 0, oor = 0;
    if (!v->pair_table || !v->ranks) {
        *out_of_range = 0;
        return 0;
    }
    size_t n_merges = 0;
    for (size_t i = 0; i <= v->pair_mask; i++)
        if (v->pair_table[i] & (1ull << 63)) n_merges++;
    for (size_t i = 0; i <= v->pair_mask; i++) {
        uint64_t e = v->pair_table[i];
        if (!(e & (1ull << 63))) continue;
        uint32_t merged = (uint32_t)(e & 0x1FFFFFu);
        uint32_t prio = tsimd_vocab_priority(v, merged);
        if (prio >= n_merges) {
            oor++;
            continue;
        }
        if (v->rank_to_id && v->rank_to_id[prio] != merged) ties++;
    }
    *out_of_range = oor;
    return ties;
}

static int check_model(const char *path, tsimd_model_format_t fmt,
                       const char *label, int want_ties) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_CL100K;
    if (tsimd_tokenizer_load_model(&t, fmt, path, NULL, &scheme, NULL, 0) !=
        0) {
        fprintf(stderr, "FAIL: cannot load %s\n", path);
        g_fail = 1;
        return -1;
    }
    const tsimd_vocab_t *v = &t.vocab;
    size_t oor = 0;
    size_t ties = count_ties(v, &oor);
    printf("  %s: vocab %u, ties %zu (prio >= n_merges: %zu)%s\n", label,
           v->vocab_size, ties, oor, want_ties ? " (ties required)" : "");
    if (want_ties && ties == 0) {
        fprintf(stderr, "FAIL %s: no tied priorities (the gate is vacuous)\n",
                label);
        g_fail = 1;
    }
    chk(v->mgr_ids != NULL, "id array built", v->mgr_ids != NULL, 1);

    size_t agree = 0, merges = 0;
    int stop = 0;

    for (uint32_t id = 0; id < v->vocab_size && !stop; id++) {
        uint32_t len;
        const uint8_t *b = tsimd_vocab_bytes(v, id, &len);
        if (len < 2 || len > 64) continue;
        for (uint32_t i = 1; i < len; i++) {
            uint32_t l = tsimd_vocab_find(v, b, i);
            uint32_t r = tsimd_vocab_find(v, b + i, len - i);
            if (l == UINT32_MAX || r == UINT32_MAX) continue;
            uint32_t pid = 0xDEADBEEFu;
            uint32_t pp = tsimd_vocab_merge_prio_id(v, l, r, &pid);
            uint32_t op = tsimd_vocab_merge_priority(v, l, r);
            uint32_t oid = tsimd_vocab_merge_id(v, l, r);
            if (pp != op) {
                chk(0, "real priority == oracle", pp, op);
                stop = 1;
                break;
            }
            if (oid == UINT32_MAX) {
                if (pp != UINT32_MAX || pid != 0xDEADBEEFu) {
                    chk(0, "real miss id untouched", pid, 0xDEADBEEFu);
                    stop = 1;
                    break;
                }
                continue;
            }
            merges++;
            if (pid != oid) {
                fprintf(stderr,
                        "FAIL %s: id mismatch l=%u r=%u probe=%u oracle=%u "
                        "prio=%u\n", label, l, r, pid, oid, pp);
                g_fail = 1;
                stop = 1;
                break;
            }
            agree++;
        }
    }

    for (int s = 0; s < 200000 && !stop; s++) {
        uint32_t l = (uint32_t)(rnd() % v->vocab_size);
        uint32_t r = (uint32_t)(rnd() % v->vocab_size);
        uint32_t pid = 0xDEADBEEFu;
        uint32_t pp = tsimd_vocab_merge_prio_id(v, l, r, &pid);
        uint32_t op = tsimd_vocab_merge_priority(v, l, r);
        uint32_t oid = tsimd_vocab_merge_id(v, l, r);
        if (pp != op || (oid == UINT32_MAX && pp != UINT32_MAX)) {
            fprintf(stderr, "FAIL %s: sample %u/%u probe %u/%u oracle %u/%u\n",
                    label, l, r, pp, pid, op, oid);
            g_fail = 1;
            stop = 1;
            break;
        }
        if (oid != UINT32_MAX && pid != oid) {
            fprintf(stderr, "FAIL %s: sample id %u/%u probe %u oracle %u\n",
                    label, l, r, pid, oid);
            g_fail = 1;
            stop = 1;
            break;
        }
        if (oid != UINT32_MAX) merges++;
        agree++;
    }
    printf("  %s: %zu pairs agreed, %zu merges\n", label, agree, merges);
    chk(agree > 1000, "enough pairs checked", agree, 1000);
    tsimd_tokenizer_free(&t);
    return 0;
}

int main(void) {
    check_synth();

    check_model(MISTRAL_MODEL, TSIMD_MODEL_SENTENCEPIECE, "mistral7b", 1);
    check_model(GEMMA_MODEL, TSIMD_MODEL_TOKENIZER_JSON, "gemma4", 0);

    check_model(QWEN_MODEL, TSIMD_MODEL_TOKENIZER_JSON, "qwen2.5", 0);
    check_model(SMOLLM_MODEL, TSIMD_MODEL_TOKENIZER_JSON, "smollm2", 0);
    printf("%s: %d checks, %d failures\n", g_fail ? "FAIL" : "PASS", checks,
           g_fail);
    return g_fail;
}
