#include "tokenizer.h"
#include "bpe.h"
#include "vocabulary.h"
#include "unicode.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MISTRAL_MODEL \
    TSIMD_SRC_DIR "/models/reference/mistral7b_tokenizer.model"

static int g_fail;
static int checks;

static void chk(int cond, const char *what, unsigned long a, unsigned long b) {
    checks++;
    if (!cond) {
        fprintf(stderr, "FAIL %s: %lu vs %lu\n", what, a, b);
        g_fail = 1;
    }
}

static uint64_t rng_state = 0xC0FFEE123456789ull;
static uint64_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static uint32_t naive_spm(const tsimd_vocab_t *v, const uint8_t *bytes,
                          uint32_t len, uint32_t *out, uint32_t cap) {
    uint32_t ids[8192];
    uint32_t n = 0;
    if (len > 8192) return 0;
    for (uint32_t i = 0; i < len;) {
        uint32_t cp;
        tsimd_char_class_t cls;
        size_t a = tsimd_utf8_next(bytes + i, len - i, &cp, &cls);
        if (a == 0) a = 1;
        uint32_t p = a == 1 ? v->byte_ids[bytes[i]]
                            : tsimd_vocab_find(v, bytes + i, (uint32_t)a);
        if (p != UINT32_MAX) {
            ids[n++] = p;
        } else {
            for (uint32_t b = 0; b < (uint32_t)a; b++)
                ids[n++] = v->byte_ids[bytes[i + b]];
        }
        i += (uint32_t)a;
    }
    for (;;) {
        uint32_t bestp = UINT32_MAX, besti = 0, bestid = 0;
        for (uint32_t i = 0; i + 1 < n; i++) {
            uint32_t id = 0;
            uint32_t p = tsimd_vocab_merge_prio_id(v, ids[i], ids[i + 1],
                                                   &id);
            if (p == UINT32_MAX) continue;
            if (p < bestp || (p == bestp && i < besti)) {
                bestp = p;
                besti = i;
                bestid = id;
            }
        }
        if (bestp == UINT32_MAX) break;
        ids[besti] = bestid;
        memmove(&ids[besti + 1], &ids[besti + 2],
                (size_t)(n - besti - 2) * sizeof(uint32_t));
        n--;
    }
    uint32_t nout = n < cap ? n : cap;
    memcpy(out, ids, (size_t)nout * sizeof(uint32_t));
    return nout;
}

static uint32_t g_scan;
static uint32_t g_queue;
static uint32_t g_bqgp;

static int encode_arm(const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
                      const uint8_t *bytes, uint32_t len, uint32_t limit,
                      uint32_t *out, uint32_t cap, uint32_t pop_probe) {
    tsimd_bpe_set_heap_linear(limit);
    tsimd_bpe_set_pop_probe(pop_probe);
    tsimd_bpe_set_pop_scan(g_scan);
    tsimd_bpe_set_queue(g_queue);
    tsimd_bpe_set_bqgp(g_bqgp);
    ws->out_count = 0;
    int rc = tsimd_bpe_encode_spm(v, ws, bytes, len);
    if (rc != 0) return -1;
    if (ws->out_count > cap) return -1;
    if (out) memcpy(out, ws->out, ws->out_count * sizeof(uint32_t));
    return (int)ws->out_count;
}

static void compare_arms(const tsimd_vocab_t *v, tsimd_bpe_workspace_t *ws,
                         const uint8_t *bytes, uint32_t len,
                         const char *label, uint32_t linear_limit) {
    uint32_t ref[16384], lin[16384], cand[16384], probe[16384];
    uint32_t scan1[16384], scan2[16384], bq1[16384], bq2[16384];
    uint32_t gp1[16384], gp2[16384];
    uint32_t nref = naive_spm(v, bytes, len, ref, 16384);
    int nl = encode_arm(v, ws, bytes, len, linear_limit, lin, 16384, 0);
    int nc = encode_arm(v, ws, bytes, len, 0, cand, 16384, 0);
    int np = encode_arm(v, ws, bytes, len, 0, probe, 16384, 1);
    g_scan = 1;
    int ns1 = encode_arm(v, ws, bytes, len, 0, scan1, 16384, 0);
    g_scan = 2;
    int ns2 = encode_arm(v, ws, bytes, len, 0, scan2, 16384, 0);
    g_scan = 0;
    g_queue = 1;
    int nb1 = encode_arm(v, ws, bytes, len, 0, bq1, 16384, 0);
    int nb2 = encode_arm(v, ws, bytes, len, 0, bq2, 16384, 0);
    g_bqgp = 1;
    int ng1 = encode_arm(v, ws, bytes, len, 0, gp1, 16384, 0);
    int ng2 = encode_arm(v, ws, bytes, len, 0, gp2, 16384, 0);
    g_bqgp = 0;
    g_queue = 0;
    if (nl < 0 || nc < 0 || np < 0 || ns1 < 0 || ns2 < 0 || nb1 < 0 ||
        nb2 < 0 || ng1 < 0 || ng2 < 0) {
        fprintf(stderr, "FAIL %s: encode rc (%d, %d, %d, %d, %d, %d, %d)\n",
                label, nl, nc, np, ns1, ns2, nb1, nb2);
        g_fail = 1;
        return;
    }
    chk((uint32_t)nl == nref && memcmp(lin, ref, nref * 4) == 0,
        "linear == oracle", (unsigned long)nl, (unsigned long)nref);
    chk((uint32_t)nc == nref && memcmp(cand, ref, nref * 4) == 0,
        "cand heap == oracle", (unsigned long)nc, (unsigned long)nref);
    chk((uint32_t)np == nref && memcmp(probe, ref, nref * 4) == 0,
        "probe heap == oracle", (unsigned long)np, (unsigned long)nref);
    chk((uint32_t)ns1 == nref && memcmp(scan1, ref, nref * 4) == 0,
        "scan1 heap == oracle", (unsigned long)ns1, (unsigned long)nref);
    chk((uint32_t)ns2 == nref && memcmp(scan2, ref, nref * 4) == 0,
        "scan2 heap == oracle", (unsigned long)ns2, (unsigned long)nref);
    chk((uint32_t)nb1 == nref && memcmp(bq1, ref, nref * 4) == 0,
        "bucket queue == oracle", (unsigned long)nb1, (unsigned long)nref);
    chk(nb1 == nb2 && memcmp(bq1, bq2, (size_t)nb1 * 4) == 0,
        "bucket queue repeatable", (unsigned long)nb1, (unsigned long)nb2);
    chk(nb1 == nc && memcmp(bq1, cand, (size_t)nb1 * 4) == 0,
        "bucket queue == cand heap", (unsigned long)nb1, (unsigned long)nc);
    chk((uint32_t)ng1 == nref && memcmp(gp1, ref, nref * 4) == 0,
        "group bucket == oracle", (unsigned long)ng1, (unsigned long)nref);
    chk(ng1 == ng2 && memcmp(gp1, gp2, (size_t)ng1 * 4) == 0,
        "group bucket repeatable", (unsigned long)ng1, (unsigned long)ng2);
    chk(ng1 == nb1 && memcmp(gp1, bq1, (size_t)ng1 * 4) == 0,
        "group bucket == per-rank bucket", (unsigned long)ng1,
        (unsigned long)nb1);
    chk(nc == np && memcmp(cand, probe, (size_t)nc * 4) == 0,
        "cand heap == probe heap", (unsigned long)nc, (unsigned long)np);
    chk(nl == nc && memcmp(lin, cand, (size_t)nl * 4) == 0,
        "linear == cand heap", (unsigned long)nl, (unsigned long)nc);
    chk(nl == np && memcmp(lin, probe, (size_t)nl * 4) == 0,
        "linear == probe heap", (unsigned long)nl, (unsigned long)np);
}

static int build_tie_vocab_ex(tsimd_vocab_t *v, uint32_t rank256) {
    memset(v, 0, sizeof *v);
    if (tsimd_vocab_build_start(v) != 0) return -1;
    for (int b = 0; b < 256; b++) {
        uint8_t c = (uint8_t)b;
        if (tsimd_vocab_add_token(v, &c, 1, (uint32_t)b) == UINT32_MAX)
            return -1;
    }
    if (tsimd_vocab_add_token(v, (const uint8_t *)"ab", 2, 256) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"ba", 2, 257) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"aba", 3, 258) ==
        UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"bab", 3, 259) ==
        UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"abc", 3, 260) ==
        UINT32_MAX)
        return -1;
    if (tsimd_vocab_seal(v, 5) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'a', 'b', 256) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'b', 'a', 257) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 256, 'a', 258) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 257, 'b', 259) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'a', 257, 260) != 0) return -1;
    tsimd_vocab_set_rank(v, 256, rank256);
    tsimd_vocab_set_rank(v, 257, 0);
    tsimd_vocab_set_rank(v, 258, 1);
    tsimd_vocab_set_rank(v, 259, 1);
    tsimd_vocab_set_rank(v, 260, 2);
    if (tsimd_vocab_pair_rank_build(v) != 0) return -1;
    return 0;
}

static int build_tie_vocab(tsimd_vocab_t *v) {
    return build_tie_vocab_ex(v, 0);
}

static void check_tie_vocab(void) {
    tsimd_vocab_t v;
    if (build_tie_vocab(&v) != 0) {
        fprintf(stderr, "FAIL: tie vocab build\n");
        g_fail = 1;
        return;
    }
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, 16) != 0) {
        g_fail = 1;
        tsimd_vocab_free(&v);
        return;
    }
    static const char *CASES[] = {"abab", "baba", "ababa", "babab", "abba",
                                  "baab", "ababab", "bababa", "abababa",
                                  "aaabbbaaa", "abcabc", "abcbac"};
    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++) {
        const uint8_t *b = (const uint8_t *)CASES[i];
        compare_arms(&v, &ws, b, (uint32_t)strlen(CASES[i]), "ties", 4096);
    }
    tsimd_bpe_workspace_free(&ws);
    tsimd_vocab_free(&v);
}

static int build_neighbor_vocab(tsimd_vocab_t *v) {
    memset(v, 0, sizeof *v);
    if (tsimd_vocab_build_start(v) != 0) return -1;
    for (int b = 0; b < 256; b++) {
        uint8_t c = (uint8_t)b;
        if (tsimd_vocab_add_token(v, &c, 1, (uint32_t)b) == UINT32_MAX)
            return -1;
    }
    if (tsimd_vocab_add_token(v, (const uint8_t *)"ab", 2, 256) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"bc", 2, 257) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"abc", 3, 258) ==
        UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"de", 2, 259) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"ef", 2, 260) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"cb", 2, 261) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"abcd", 4, 262) ==
        UINT32_MAX)
        return -1;
    if (tsimd_vocab_seal(v, 7) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'a', 'b', 256) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'b', 'c', 257) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'a', 257, 258) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'd', 'e', 259) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'e', 'f', 260) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'c', 'b', 261) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 258, 'd', 262) != 0) return -1;
    tsimd_vocab_set_rank(v, 256, 1);
    tsimd_vocab_set_rank(v, 257, 0);
    tsimd_vocab_set_rank(v, 258, 1);
    tsimd_vocab_set_rank(v, 259, 5);
    tsimd_vocab_set_rank(v, 260, 0);
    tsimd_vocab_set_rank(v, 261, 1);
    tsimd_vocab_set_rank(v, 262, 1);
    if (tsimd_vocab_pair_rank_build_slots(v) != 0) return -1;
    return 0;
}

static void check_bucket_tie(void) {
    tsimd_vocab_t v;
    if (build_tie_vocab(&v) != 0) {
        fprintf(stderr, "FAIL: bucket tie vocab build\n");
        g_fail = 1;
        return;
    }
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, 16) != 0) {
        g_fail = 1;
        tsimd_vocab_free(&v);
        return;
    }
    uint32_t out[64];
    uint32_t runs_before = tsimd_bpe_bucket_runs();
    g_queue = 1;
    int n1 = encode_arm(&v, &ws, (const uint8_t *)"abab", 4, 0, out, 64, 0);
    chk(n1 == 2 && out[0] == 256 && out[1] == 256,
        "bucket stale skip abab == 256,256",
        (unsigned long)(n1 > 0 ? out[0] : 0), 256);
    int n2 = encode_arm(&v, &ws, (const uint8_t *)"baba", 4, 0, out, 64, 0);
    chk(n2 == 2 && out[0] == 257 && out[1] == 257,
        "bucket stale skip baba == 257,257",
        (unsigned long)(n2 > 0 ? out[0] : 0), 257);
    int n3 = encode_arm(&v, &ws, (const uint8_t *)"ababa", 5, 0, out, 64, 0);
    chk(n3 == 2 && out[0] == 256 && out[1] == 258,
        "bucket tie ababa == 256,258",
        (unsigned long)(n3 > 0 ? out[0] : 0), 256);
    uint32_t runs_after = tsimd_bpe_bucket_runs();
    chk(runs_after == runs_before + 3, "bucket arm really ran 3 times",
        (unsigned long)runs_after - runs_before, 3);
    g_queue = 0;
    int n4 = encode_arm(&v, &ws, (const uint8_t *)"abab", 4, 0, out, 64, 0);
    chk(n4 == 2 && out[0] == 256 && out[1] == 256,
        "heap stale skip abab == 256,256",
        (unsigned long)(n4 > 0 ? out[0] : 0), 256);
    int n5 = encode_arm(&v, &ws, (const uint8_t *)"ababa", 5, 0, out, 64, 0);
    chk(n5 == 2 && out[0] == 256 && out[1] == 258,
        "heap tie ababa == 256,258",
        (unsigned long)(n5 > 0 ? out[0] : 0), 256);
    uint32_t ref[64];
    uint32_t nref = naive_spm(&v, (const uint8_t *)"ababa", 5, ref, 64);
    chk(nref == 2 && ref[0] == 256 && ref[1] == 258,
        "oracle tie ababa == 256,258",
        (unsigned long)(nref > 0 ? ref[0] : 0), 256);
    uint32_t nref2 = naive_spm(&v, (const uint8_t *)"abab", 4, ref, 64);
    chk(nref2 == 2 && ref[0] == 256 && ref[1] == 256,
        "oracle stale skip abab == 256,256",
        (unsigned long)(nref2 > 0 ? ref[0] : 0), 256);
    tsimd_bpe_workspace_free(&ws);
    tsimd_vocab_free(&v);
}

static int build_order_vocab(tsimd_vocab_t *v) {
    memset(v, 0, sizeof *v);
    if (tsimd_vocab_build_start(v) != 0) return -1;
    for (int b = 0; b < 256; b++) {
        uint8_t c = (uint8_t)b;
        if (tsimd_vocab_add_token(v, &c, 1, (uint32_t)b) == UINT32_MAX)
            return -1;
    }
    if (tsimd_vocab_add_token(v, (const uint8_t *)"ab", 2, 256) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"cd", 2, 257) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"bc", 2, 258) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_add_token(v, (const uint8_t *)"abc", 3, 259) == UINT32_MAX)
        return -1;
    if (tsimd_vocab_seal(v, 5) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'a', 'b', 256) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'c', 'd', 257) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 'b', 'c', 258) != 0) return -1;
    if (tsimd_vocab_record_merge(v, 256, 'c', 259) != 0) return -1;
    tsimd_vocab_set_rank(v, 256, 0);
    tsimd_vocab_set_rank(v, 257, 0);
    tsimd_vocab_set_rank(v, 258, 1);
    tsimd_vocab_set_rank(v, 259, 0);
    if (tsimd_vocab_pair_rank_build(v) != 0) return -1;
    return 0;
}

static void check_bucket_order_minpos(void) {
    tsimd_vocab_t v;
    if (build_order_vocab(&v) != 0) {
        fprintf(stderr, "FAIL: order vocab build\n");
        g_fail = 1;
        return;
    }
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, 16) != 0) {
        g_fail = 1;
        tsimd_vocab_free(&v);
        return;
    }
    static const char *CASES[] = {"abcd", "abcdcd", "abcdabcd", "cdabcd"};
    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++) {
        const uint8_t *b = (const uint8_t *)CASES[i];
        uint32_t n = (uint32_t)strlen(CASES[i]);
        uint32_t ref[64], gp[64];
        uint32_t nref = naive_spm(&v, b, n, ref, 64);
        g_queue = 1;
        g_bqgp = 1;
        int ng = encode_arm(&v, &ws, b, n, 0, gp, 64, 0);
        g_bqgp = 0;
        g_queue = 0;
        char what[64];
        snprintf(what, sizeof what, "bucket minpos %s", CASES[i]);
        chk(ng == (int)nref && memcmp(gp, ref, (size_t)nref * 4) == 0, what,
            (unsigned long)(ng > 0 ? gp[0] : 0),
            (unsigned long)(nref > 0 ? ref[0] : 0));
    }
    tsimd_bpe_workspace_free(&ws);
    tsimd_vocab_free(&v);
}

static void check_bucket_rank_bound(void) {
    tsimd_vocab_t v;
    if (build_tie_vocab_ex(&v, 100000) != 0) {
        fprintf(stderr, "FAIL: rank bound vocab build\n");
        g_fail = 1;
        return;
    }
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, 16) != 0) {
        g_fail = 1;
        tsimd_vocab_free(&v);
        return;
    }
    uint32_t ref[64], out[64];
    uint32_t nref = naive_spm(&v, (const uint8_t *)"abab", 4, ref, 64);
    uint32_t before = tsimd_bpe_bucket_runs();
    g_queue = 1;
    int n = encode_arm(&v, &ws, (const uint8_t *)"abab", 4, 0, out, 64, 0);
    uint32_t after = tsimd_bpe_bucket_runs();
    g_queue = 0;
    chk(n == (int)nref && memcmp(out, ref, (size_t)nref * 4) == 0,
        "rank out of range falls back to the heap",
        (unsigned long)(n > 0 ? out[0] : 0),
        (unsigned long)(nref > 0 ? ref[0] : 0));
    chk(after == before, "rank out of range did not complete a bucket run",
        (unsigned long)(after - before), 0);
    tsimd_bpe_workspace_free(&ws);
    tsimd_vocab_free(&v);
}

static void check_shift_trap(void) {
    tsimd_vocab_t v;
    if (build_neighbor_vocab(&v) != 0) {
        fprintf(stderr, "FAIL: shift trap vocab build\n");
        g_fail = 1;
        return;
    }
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, 16) != 0) {
        g_fail = 1;
        tsimd_vocab_free(&v);
        return;
    }
    uint32_t id1 = 0, id2 = 0;
    uint32_t p1 = tsimd_vocab_merge_prio_id(&v, 'a', 'b', &id1);
    uint32_t p2 = tsimd_vocab_merge_prio_id(&v, 'a', 257, &id2);
    chk(p1 == p2 && id1 != id2, "trap pairs share rank, differ id", id1, id2);
    static const char *CASES[] = {"abca", "abcac", "abcdd", "abcdab"};
    for (size_t k = 0; k < sizeof(CASES) / sizeof(CASES[0]); k++) {
        const uint8_t *b = (const uint8_t *)CASES[k];
        uint32_t n = (uint32_t)strlen(CASES[k]);
        uint32_t ref[16384], cand[16384], probe[16384];
        uint32_t nref = naive_spm(&v, b, n, ref, 16384);
        int nc = encode_arm(&v, &ws, b, n, 0, cand, 16384, 0);
        int np = encode_arm(&v, &ws, b, n, 0, probe, 16384, 1);
        char what[64];
        snprintf(what, sizeof what, "shift trap %s cand", CASES[k]);
        chk(nc == (int)nref && memcmp(cand, ref, (size_t)nref * 4) == 0, what,
            (unsigned long)(nc > 0 ? cand[0] : 0),
            (unsigned long)(nref > 0 ? ref[0] : 0));
        snprintf(what, sizeof what, "shift trap %s probe", CASES[k]);
        chk(np == (int)nref && memcmp(probe, ref, (size_t)nref * 4) == 0, what,
            (unsigned long)(np > 0 ? probe[0] : 0),
            (unsigned long)(nref > 0 ? ref[0] : 0));
    }
    tsimd_bpe_workspace_free(&ws);
    tsimd_vocab_free(&v);
}

static void check_neighbor_vocab(void) {
    tsimd_vocab_t v;
    if (build_neighbor_vocab(&v) != 0) {
        fprintf(stderr, "FAIL: neighbor vocab build\n");
        g_fail = 1;
        return;
    }
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, 16) != 0) {
        g_fail = 1;
        tsimd_vocab_free(&v);
        return;
    }
    static const char *CASES[] = {
        "abc",    "abcb",   "abcbc",  "cbabc", "abbc",  "abcabc",
        "def",    "defdef", "ddef",   "edef",  "abcdef", "abab",
        "bcbc",   "cbcb",   "cbab",   "deffed", "abcdeffedcba",
        "abcdcba"};
    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++) {
        const uint8_t *b = (const uint8_t *)CASES[i];
        compare_arms(&v, &ws, b, (uint32_t)strlen(CASES[i]), "neighbor", 4096);
    }
    for (int k = 0; k < 1000; k++) {
        uint8_t b[40];
        uint32_t n = 1 + (uint32_t)(rnd() % 32);
        for (uint32_t i = 0; i < n; i++)
            b[i] = (uint8_t)('a' + (rnd() % 6));
        compare_arms(&v, &ws, b, n, "neighbor random", 4096);
    }
    tsimd_bpe_workspace_free(&ws);
    tsimd_vocab_free(&v);
}

#define NWORDS 20
static const char *WORDS[NWORDS] = {
    "the",   "quick", "brown",     "fox",      "jumps",  "over",  "lazy",
    "dog",   "token", "pipeline",  "throughput", "and",  "of",    "to",
    "in",    "a",     "sentence",  "piece",    "merge",  "heap"
};

static size_t gen_text(uint8_t *dst, size_t cap) {
    size_t n = 0;
    while (n + 24 < cap) {
        const char *w = WORDS[rnd() % NWORDS];
        size_t wl = strlen(w);
        memcpy(dst + n, w, wl);
        n += wl;
        dst[n++] = (rnd() % 9 == 0) ? '\n' : ' ';
    }
    return n;
}

static void check_mistral_paths(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_SPM;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_SENTENCEPIECE, MISTRAL_MODEL,
                                   NULL, &scheme, NULL, 0) != 0) {
        fprintf(stderr, "FAIL: cannot load %s\n", MISTRAL_MODEL);
        g_fail = 1;
        return;
    }
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, t.vocab.max_token_len) != 0) {
        g_fail = 1;
        tsimd_tokenizer_free(&t);
        return;
    }
    static const size_t LENS[] = {2, 3, 7, 16, 32, 63, 64, 65, 96, 128, 200,
                                  333};
    uint8_t raw[4096];
    uint8_t norm[12288];
    for (size_t li = 0; li < sizeof(LENS) / sizeof(LENS[0]); li++) {
        size_t want = LENS[li];
        size_t n = gen_text(raw, want);
        size_t nn = 0;
        for (size_t i = 0; i < n; i++) {
            if (raw[i] == ' ') {
                norm[nn++] = 0xE2;
                norm[nn++] = 0x96;
                norm[nn++] = 0x81;
            } else {
                norm[nn++] = raw[i];
            }
        }
        compare_arms(&t.vocab, &ws, norm, (uint32_t)nn, "mistral paths", 1u
                                                                    << 20);
    }
    tsimd_bpe_workspace_free(&ws);
    tsimd_tokenizer_free(&t);
}

static void check_pair_tables(const tsimd_vocab_t *a, const tsimd_vocab_t *b,
                              const char *label) {
    unsigned long n_pairs = 0, n_bad = 0;
    for (uint32_t id = 0; id < a->vocab_size; id++) {
        uint32_t len = a->offsets[id + 1] - a->offsets[id];
        if (len < 2) continue;
        const uint8_t *bytes = a->bytes + a->offsets[id];
        for (uint32_t w = 1; w < len; w++) {
            uint32_t left = w == 1 ? a->byte_ids[bytes[0]]
                                   : tsimd_vocab_find(a, bytes, w);
            uint32_t right = (len - w) == 1 ? a->byte_ids[bytes[w]]
                                            : tsimd_vocab_find(a, bytes + w,
                                                               len - w);
            if (left == UINT32_MAX || right == UINT32_MAX) continue;
            uint32_t ia = 0, ib = 0, fa = 0;
            uint32_t pa = tsimd_vocab_merge_prio_id(a, left, right, &ia);
            uint32_t pb = tsimd_vocab_merge_prio_id(b, left, right, &ib);
            uint32_t pf = tsimd_vocab_merge_prio_id_fast(a, left, right, &fa);
            uint32_t qa = tsimd_vocab_merge_priority(a, left, right);
            uint32_t qb = tsimd_vocab_merge_priority(b, left, right);
            uint32_t qf = tsimd_vocab_merge_priority_fast(a, left, right);
            n_pairs++;
            if (pa != pb || ia != ib || pa != pf || fa != ia || pa != qa ||
                pb != qb || qa != qf) {
                if (n_bad < 5)
                    fprintf(stderr,
                            "FAIL %s pair (%u,%u): prio %u/%u fast %u, id "
                            "%u/%u fast %u\n",
                            label, left, right, pa, pb, pf, ia, ib, fa);
                n_bad++;
            }
        }
    }
    chk(n_bad == 0, label, n_bad, n_pairs);
    printf("  %s: %lu pairs compared, %lu mismatches\n", label, n_pairs,
           n_bad);
    unsigned long n_miss = 0;
    for (int k = 0; k < 20000; k++) {
        uint32_t l = (uint32_t)(rnd() % a->vocab_size);
        uint32_t r = (uint32_t)(rnd() % a->vocab_size);
        uint32_t ia = 123, ib = 456, fa = 789;
        uint32_t pa = tsimd_vocab_merge_prio_id(a, l, r, &ia);
        uint32_t pb = tsimd_vocab_merge_prio_id(b, l, r, &ib);
        uint32_t pf = tsimd_vocab_merge_prio_id_fast(a, l, r, &fa);
        uint32_t qa = tsimd_vocab_merge_priority(a, l, r);
        uint32_t qf = tsimd_vocab_merge_priority_fast(a, l, r);
        if (pa != pb || pa != pf || pa != qa || qa != qf) n_miss++;
        if (pa != UINT32_MAX && (ia != ib || fa != ia)) n_miss++;
    }
    chk(n_miss == 0, "random pairs agree", n_miss, 20000);
}

static int load_mistral(tsimd_vocab_t *v) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_SPM;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_SENTENCEPIECE, MISTRAL_MODEL,
                                   NULL, &scheme, NULL, 0) != 0)
        return -1;
    *v = t.vocab;
    memset(&t.vocab, 0, sizeof t.vocab);
    tsimd_tokenizer_free(&t);
    return 0;
}

static void check_dense_tables(void) {
    tsimd_vocab_t slots, dense, dlog9;
    if (load_mistral(&slots) != 0) {
        fprintf(stderr, "FAIL: mistral load (default)\n");
        g_fail = 1;
        return;
    }
    setenv("TSIMD_MGR_DENSE", "1", 1);
    if (load_mistral(&dense) != 0) {
        fprintf(stderr, "FAIL: mistral load (dense forced)\n");
        g_fail = 1;
        goto out_slots;
    }
    setenv("TSIMD_MGR_DLOG", "9", 1);
    if (load_mistral(&dlog9) != 0) {
        fprintf(stderr, "FAIL: mistral load (dlog9)\n");
        g_fail = 1;
        goto out_dense;
    }
    unsetenv("TSIMD_MGR_DLOG");
    unsetenv("TSIMD_MGR_DENSE");
    chk(slots.mgr_dense == NULL, "spm-BPE default skips the dense grid",
        slots.mgr_dense == NULL, 0);
    chk(slots.mgr_slots != NULL, "slots built", slots.mgr_slots != NULL, 0);
    chk(dense.mgr_dense != NULL, "dense grid forced on",
        dense.mgr_dense != NULL, 0);
    chk(dense.mgr_dense_ids != NULL, "dense ids forced on",
        dense.mgr_dense_ids != NULL, 0);
    chk(dlog9.mgr_dense != NULL && dlog9.mgr_dense_log2 == 9, "dlog 9",
        dlog9.mgr_dense_log2, 9);
    check_pair_tables(&dense, &slots, "dense grid vs slots");
    check_pair_tables(&dlog9, &slots, "dlog9 grid vs slots");
    tsimd_vocab_free(&dlog9);
out_dense:
    tsimd_vocab_free(&dense);
out_slots:
    tsimd_vocab_free(&slots);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void ladder(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_SPM;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_SENTENCEPIECE, MISTRAL_MODEL,
                                   NULL, &scheme, NULL, 0) != 0)
        return;
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, t.vocab.max_token_len) != 0) return;
    printf("ladder (linear vs heap, one arm each, symbols = normalized bytes):"
           "\n");
    static const uint32_t NS[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    uint8_t raw[8192];
    static uint8_t norm[32768];
    for (size_t k = 0; k < sizeof(NS) / sizeof(NS[0]); k++) {
        uint32_t want = NS[k];
        size_t n = gen_text(raw, want);
        size_t nn = 0;
        for (size_t i = 0; i < n && nn + 4 < sizeof(norm); i++) {
            if (raw[i] == ' ') {
                norm[nn++] = 0xE2;
                norm[nn++] = 0x96;
                norm[nn++] = 0x81;
            } else {
                norm[nn++] = raw[i];
            }
        }
        uint32_t sym = 0;
        for (size_t i = 0; i < nn;) {
            uint32_t cp;
            tsimd_char_class_t cls;
            size_t a = tsimd_utf8_next(norm + i, nn - i, &cp, &cls);
            if (a == 0) a = 1;
            sym++;
            i += (size_t)a;
        }
        double tl = 0, th = 0;
        uint32_t lin_out[16384], heap_out[16384];
        int nl = encode_arm(&t.vocab, &ws, norm, (uint32_t)nn, 1u << 20,
                            lin_out, 16384, 0);
        int nh = encode_arm(&t.vocab, &ws, norm, (uint32_t)nn, 0, heap_out,
                            16384, 0);
        chk(nl == nh && nl > 0 &&
                memcmp(lin_out, heap_out, (size_t)nl * 4) == 0,
            "ladder arms agree", (unsigned long)nl, (unsigned long)nh);
        int reps = sym <= 512 ? 40 : 8;
        for (int r = 0; r < reps; r++) {
            double s = now_sec();
            encode_arm(&t.vocab, &ws, norm, (uint32_t)nn, 1u << 20, NULL, 0,
                       0);
            tl += now_sec() - s;
            s = now_sec();
            encode_arm(&t.vocab, &ws, norm, (uint32_t)nn, 0, NULL, 0, 0);
            th += now_sec() - s;
        }
        printf("  syms %5u (%6zu bytes): linear %9.3f us  heap %9.3f us  "
               "linear/heap %.2f\n",
               sym, nn, tl / reps * 1e6, th / reps * 1e6, tl / th);
    }
    tsimd_bpe_set_heap_linear(64);
    tsimd_bpe_workspace_free(&ws);
    tsimd_tokenizer_free(&t);
}

int main(void) {
    check_bucket_order_minpos();
    check_tie_vocab();
    check_bucket_tie();
    check_bucket_rank_bound();
    check_shift_trap();
    check_neighbor_vocab();
    check_mistral_paths();
    check_dense_tables();
    ladder();
    if (g_fail) {
        fprintf(stderr, "test_bpe_path_agreement: FAILED (%d checks)\n",
                checks);
        return 1;
    }
    printf("test_bpe_path_agreement: OK (%d checks)\n", checks);
    return 0;
}
