#include "tokensimd_sys.h"

#include "allocator.h"
#include "tokenizer.h"
#include "batch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL100K TSIMD_SRC_DIR "/models/cl100k_base.tiktoken"
#define T5 TSIMD_SRC_DIR "/models/reference/t5_spiece.model"
#define QWEN TSIMD_SRC_DIR "/models/reference/qwen2.5-0.5b_tokenizer.json"

static int fails = 0;

static void check(int cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

int main(void) {
    char err[256];

    {
        uint8_t *p1 = tsimd_xmalloc(3u << 20);
        check(p1 != NULL, "xmalloc 3MiB");
        memset(p1, 0xAB, 3u << 20);
        tsimd_xfree(p1);
        uint8_t *p2 = tsimd_xmalloc(2u << 20);
        check(p2 != NULL, "xmalloc 2MiB after the big free");
        memset(p2, 0xCD, 2u << 20);
        check(((uintptr_t)p2 & 15u) == 0, "xmalloc 16-byte alignment");
        uint8_t *p3 = tsimd_xmalloc(1u << 20);
        check(p3 != NULL, "xmalloc 1MiB");
        tsimd_xfree(p3);
        tsimd_xfree(p2);
        void *z = tsimd_xmalloc(0);
        check(z != NULL, "xmalloc 0");
        tsimd_xfree(z);
        tsimd_xfree(NULL);
    }

    tsimd_sys *h = tsimd_sys_open(CL100K, "tiktoken", "cl100k", err, sizeof err);
    check(h != NULL, "open cl100k");
    if (!h) {
        fprintf(stderr, "  %s\n", err);
        return 1;
    }
    check(tsimd_sys_vocab_size(h) == 100256, "cl100k vocab size");
    check(strcmp(tsimd_sys_version(), "1.0.4") == 0,
          "version string");

    const char *text = "Hello world, this is a test.";
    int d_threads0 = tsimd_default_threads();
    uint32_t *ids = NULL;
    size_t n_ids = 0;
    check(tsimd_sys_encode(h, (const uint8_t *)text, strlen(text), 1, &ids,
                           &n_ids) == 0,
          "encode");
    check(n_ids == 8, "cl100k id count");
    uint8_t *bytes = NULL;
    size_t n_bytes = 0;
    check(tsimd_sys_decode(h, ids, n_ids, 0, &bytes, &n_bytes) == 0, "decode");
    check(n_bytes == strlen(text) &&
              memcmp(bytes, text, n_bytes) == 0,
          "decode round trip");
    tsimd_sys_free(bytes);

    size_t n_own = 0;
    const uint32_t *own = NULL;
    {
        tsimd_tokenizer_t dt;
        memset(&dt, 0, sizeof dt);
        tsimd_pretok_scheme_t dsch = TSIMD_PRETOK_CL100K;
        if (tsimd_tokenizer_load_model(&dt, TSIMD_MODEL_TIKTOKEN,
                                       TSIMD_SRC_DIR
                                       "/models/cl100k_base.tiktoken",
                                       NULL, &dsch, NULL, 0) == 0) {
            size_t need = 0;
            check(tsimd_tokenizer_decode_into(&dt, ids, n_ids, 0, NULL, 0,
                                              &need) == 0 &&
                      need == strlen(text),
                  "decode into count mode");
            uint8_t *dbuf = malloc(need);
            size_t got = 0;
            check(tsimd_tokenizer_decode_into(&dt, ids, n_ids, 0, dbuf, need,
                                              &got) == 0 && got == need &&
                      memcmp(dbuf, text, need) == 0,
                  "decode into matches");
            uint8_t *sbuf = malloc(need);
            size_t sgot = 0;
            check(tsimd_sys_decode_into(h, ids, n_ids, 0, sbuf, need,
                                        &sgot) == 0 && sgot == need &&
                      memcmp(sbuf, text, need) == 0,
                  "sys decode into matches");
            check(tsimd_sys_decode_into(h, ids, n_ids, 0, sbuf, need - 1,
                                        &sgot) != 0,
                  "sys decode into must fail on a short buffer");
            size_t pgot = 0;
            check(tsimd_sys_decode_parallel(h, ids, n_ids, 0, sbuf, need,
                                            &pgot, 0) == 0 && pgot == need &&
                      memcmp(sbuf, text, need) == 0,
                  "sys decode parallel matches");
            free(sbuf);
            check(tsimd_tokenizer_decode_into(&dt, ids, n_ids, 0, dbuf,
                                              need - 1, &got) != 0,
                  "decode into must fail on a short buffer");
            free(dbuf);
            tsimd_tokenizer_free(&dt);
        }
    }
    check(tsimd_sys_encode_own(h, (const uint8_t *)text, strlen(text), 1) == 0,
          "encode own");
    own = tsimd_sys_ids(h, &n_own);
    check(own != NULL && n_own == n_ids &&
              memcmp(own, ids, n_ids * sizeof(uint32_t)) == 0,
          "encode own ids equal the sequential ids");
    check(tsimd_sys_encode_own(h, (const uint8_t *)text, strlen(text), 4) == 0,
          "encode own parallel");
    own = tsimd_sys_ids(h, &n_own);
    check(own != NULL && n_own == n_ids &&
              memcmp(own, ids, n_ids * sizeof(uint32_t)) == 0,
          "encode own parallel ids equal the sequential ids");
    check(tsimd_sys_encode_own(h, (const uint8_t *)text, strlen(text), 0) == 0,
          "encode own automatic");
    own = tsimd_sys_ids(h, &n_own);
    check(own != NULL && n_own == n_ids &&
              memcmp(own, ids, n_ids * sizeof(uint32_t)) == 0,
          "encode own automatic ids equal the sequential ids");
    check(tsimd_sys_encode_own(h, (const uint8_t *)"", 0, 1) == 0,
          "encode own empty");
    own = tsimd_sys_ids(h, &n_own);
    check(n_own == 0, "encode own empty id count");

    uint32_t *ids4 = NULL;
    size_t n4 = 0;    check(tsimd_sys_encode(h, (const uint8_t *)text, strlen(text), 4, &ids4,
                           &n4) == 0,
          "encode parallel");
    check(n4 == n_ids && memcmp(ids4, ids, n_ids * sizeof(uint32_t)) == 0,
          "parallel ids equal the sequential ids");
    tsimd_sys_free(ids4);

    size_t bound = 0;
    check(tsimd_sys_encode_bound(h, (const uint8_t *)text, strlen(text), 1,
                                 &bound) == 0 && bound >= n_ids,
          "encode bound");
    uint32_t *into = malloc((bound ? bound : 1) * sizeof(uint32_t));
    size_t n_into = 0;
    memset(into, 0, (bound ? bound : 1) * sizeof(uint32_t));
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 1,
                                into, bound, &n_into) == 0 && n_into == n_ids,
          "encode into");
    check(memcmp(into, ids, n_ids * sizeof(uint32_t)) == 0,
          "encode into ids equal the sequential ids");
    memset(into, 0, (bound ? bound : 1) * sizeof(uint32_t));
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 4,
                                into, bound, &n_into) == 0 && n_into == n_ids,
          "encode into parallel");
    check(memcmp(into, ids, n_ids * sizeof(uint32_t)) == 0,
          "encode into parallel ids equal the sequential ids");
    uint32_t *tight = malloc(n_ids * sizeof(uint32_t));
    memset(tight, 0, n_ids * sizeof(uint32_t));
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 1,
                                tight, n_ids, &n_into) == 0 && n_into == n_ids,
          "encode into with a tight buffer");
    check(memcmp(tight, ids, n_ids * sizeof(uint32_t)) == 0,
          "tight-buffer ids equal the sequential ids");
    memset(tight, 0, n_ids * sizeof(uint32_t));
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 4,
                                tight, n_ids, &n_into) == 0 && n_into == n_ids,
          "encode into parallel with a tight buffer");
    check(memcmp(tight, ids, n_ids * sizeof(uint32_t)) == 0,
          "tight-buffer parallel ids equal the sequential ids");
    free(tight);
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 1,
                                into, 1, &n_into) != 0,
          "encode into must fail on a short buffer");
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 4,
                                into, 1, &n_into) != 0,
          "encode into parallel must fail on a short buffer");
    check(tsimd_default_threads() >= 1, "default thread count");
    memset(into, 0, (bound ? bound : 1) * sizeof(uint32_t));
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 0,
                                into, bound, &n_into) == 0 && n_into == n_ids,
          "encode into with automatic threads");
    check(memcmp(into, ids, n_ids * sizeof(uint32_t)) == 0,
          "automatic-thread ids equal the sequential ids");
    uint32_t *ids0 = NULL;
    size_t n0 = 0;
    check(tsimd_sys_encode(h, (const uint8_t *)text, strlen(text), 0, &ids0,
                           &n0) == 0 &&
              n0 == n_ids && memcmp(ids0, ids, n_ids * sizeof(uint32_t)) == 0,
          "automatic-thread encode ids equal the sequential ids");
    tsimd_sys_free(ids0);
    {
        const char *unit = "The quick brown fox jumps over the lazy dog. ";
        size_t ul = strlen(unit);
        size_t bl = (1u << 19) + 7;
        uint8_t *big = malloc(bl);
        for (size_t i = 0; i < bl; i++) big[i] = (uint8_t)unit[i % ul];
        uint32_t *b1 = NULL, *b0 = NULL, *b6 = NULL;
        size_t n1b = 0, n0b = 0, n6b = 0;
        check(tsimd_sys_encode(h, big, bl, 1, &b1, &n1b) == 0,
              "big payload sequential");
        check(tsimd_sys_encode(h, big, bl, 0, &b0, &n0b) == 0,
              "big payload automatic");
        check(tsimd_sys_encode(h, big, bl, 6, &b6, &n6b) == 0,
              "big payload 6 threads");
        check(n1b == n0b && n0b == n6b && n1b > 0 &&
                  memcmp(b1, b0, n1b * sizeof(uint32_t)) == 0 &&
                  memcmp(b1, b6, n1b * sizeof(uint32_t)) == 0,
              "big auto and 6-thread ids equal the sequential ids");
        {
            int d2 = tsimd_default_threads();
            check(d_threads0 < 2 || d2 == d_threads0,
                  "default thread count is stable across a pinned call");
        }
        tsimd_sys_free(b1);
        tsimd_sys_free(b0);
        tsimd_sys_free(b6);
        {
            const char *longer =
                "A considerably longer sentence, repeated to force the "
                "reuse buffer to grow beyond its first size. ";
            size_t nl = strlen(longer);
            uint32_t *ref = NULL;
            size_t nr = 0;
            check(tsimd_sys_encode(h, (const uint8_t *)longer, nl, 1, &ref,
                                   &nr) == 0,
                  "reference encode for the reuse checks");
            check(tsimd_sys_encode_own(h, (const uint8_t *)longer, nl, 1) == 0,
                  "encode own growth");
            own = tsimd_sys_ids(h, &n_own);
            check(own != NULL && n_own == nr &&
                      memcmp(own, ref, nr * sizeof(uint32_t)) == 0,
                  "grown-buffer ids equal the reference");
            const uint32_t *p_grown = own;
            check(tsimd_sys_encode_own(h, big, bl, 1) == 0,
                  "encode own on the big payload");
            own = tsimd_sys_ids(h, &n_own);
            check(n_own == n1b, "big own id count");
            const uint32_t *p_big = own;
            check(tsimd_sys_encode_own(h, (const uint8_t *)text,
                                       strlen(text), 1) == 0,
                  "encode own shrink");
            own = tsimd_sys_ids(h, &n_own);
            check(own == p_big, "the reuse buffer survives the smaller call");
            check(n_own == n_ids &&
                      memcmp(own, ids, n_ids * sizeof(uint32_t)) == 0,
                  "shrunk-call ids equal the sequential ids");
            tsimd_sys_free(ref);
        }
        {
            tsimd_input_span_t spans[9];
            const char *parts[9] = {"Hello, ", "world! ", "The quick ",
                                    "brown fox ", "jumps over ",
                                    "the lazy dog. ", "def fib(n):\\n",
                                    "    return n\\n", "x"};
            uint32_t *flat = NULL;
            size_t n_flat = 0;
            size_t *offs = NULL;
            for (int i = 0; i < 9; i++) {
                spans[i].data = (const uint8_t *)parts[i];
                spans[i].length = strlen(parts[i]);
            }
            check(tsimd_sys_encode_batch(h, spans, 9, 0, &flat, &n_flat,
                                         &offs) == 0,
                  "encode batch");
            check(offs != NULL && offs[0] == 0, "batch offsets start");
            size_t acc = 0;
            int all_eq = 1;
            for (int i = 0; i < 9; i++) {
                uint32_t *one = NULL;
                size_t n_one = 0;
                check(tsimd_sys_encode(h, (const uint8_t *)parts[i],
                                       strlen(parts[i]), 1, &one,
                                       &n_one) == 0,
                      "batch per-doc reference");
                if (offs[i] != acc) all_eq = 0;
                if (n_flat < offs[i] + n_one ||
                    memcmp(flat + offs[i], one,
                           n_one * sizeof(uint32_t)) != 0)
                    all_eq = 0;
                acc = offs[i] + n_one;
                tsimd_sys_free(one);
            }
            if (acc != n_flat) all_eq = 0;
            check(all_eq, "batch ids and offsets equal the per-doc encodes");
            uint32_t *eflat = NULL;
            size_t n_eflat = 0;
            size_t *eoffs = NULL;
            check(tsimd_sys_encode_batch(h, spans, 0, 0, &eflat, &n_eflat,
                                         &eoffs) == 0,
                  "encode batch empty");
            check(n_eflat == 0, "encode batch empty id count");
        }
        free(big);
    }
    free(into);
    tsimd_sys_free(ids);
    tsimd_sys_close(h);

    {
        tsimd_tokenizer_t a;
        memset(&a, 0, sizeof a);
        tsimd_pretok_scheme_t g2 = TSIMD_PRETOK_GPT2;
        tsimd_pretok_scheme_t cl = TSIMD_PRETOK_CL100K;
        check(tsimd_tokenizer_load_model(&a, TSIMD_MODEL_TIKTOKEN, CL100K,
                                         NULL, &g2, NULL, 0) == 0,
              "tiktoken auto-scheme load");
        tsimd_tokenizer_t b;
        memset(&b, 0, sizeof b);
        check(tsimd_tokenizer_load_model(&b, TSIMD_MODEL_TIKTOKEN, CL100K,
                                         NULL, &cl, NULL, 0) == 0,
              "tiktoken explicit cl100k load");
        if (a.scheme == TSIMD_PRETOK_CL100K && b.scheme == TSIMD_PRETOK_CL100K) {
            const char *sample = "The quick brown fox jumps over the lazy dog.";
            tsimd_token_sequence_t sa, sb;
            check(tsimd_tokenizer_encode(&a, (const uint8_t *)sample,
                                         strlen(sample), &sa) == 0,
                  "auto-scheme encode");
            check(tsimd_tokenizer_encode(&b, (const uint8_t *)sample,
                                         strlen(sample), &sb) == 0,
                  "explicit-scheme encode");
            check(sa.count == sb.count &&
                      memcmp(sa.ids, sb.ids, sa.count * 4) == 0,
                  "auto-scheme ids equal explicit cl100k ids");
            tsimd_token_sequence_free(&sa);
            tsimd_token_sequence_free(&sb);
        } else {
            check(0, "auto-scheme resolve wrote back cl100k");
        }
        tsimd_tokenizer_free(&a);
        tsimd_tokenizer_free(&b);
    }
    h = tsimd_sys_open(T5, "sentencepiece", "auto", err, sizeof err);
    check(h != NULL, "open t5");
    if (h) {
        check(tsimd_sys_vocab_size(h) == 32000, "t5 vocab size");
        size_t big_len = 16u << 20;
        uint8_t *big = malloc(big_len);
        check(big != NULL, "alloc t5 16MiB input");
        if (big) {
            const char *pat = "una mattina mi svegliai; comunisme, caf\xc3\xa9, "
                              "\xe6\x9d\xb1\xe4\xba\xac na\xc3\xafve "
                              "fa\xc3\xa7" "ade. ";
            size_t plen = strlen(pat);
            for (size_t i = 0; i < big_len; i++)
                big[i] = (uint8_t)pat[i % plen];
            tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
            tsimd_tokenizer_t ref;
            check(tsimd_tokenizer_load_model(&ref, TSIMD_MODEL_SENTENCEPIECE,
                                             T5, NULL, &sch, NULL, 0) == 0,
                  "load t5 direct reference");
            uint32_t *sids1 = NULL;
            size_t sn1 = 0;
            check(tsimd_sys_encode(h, big, big_len, 1, &sids1, &sn1) == 0,
                  "t5 sys encode 1t past the into cap");
            tsimd_token_sequence_t r1;
            check(tsimd_tokenizer_encode(&ref, big, big_len, &r1) == 0,
                  "t5 seq reference 1t");
            check(sn1 == r1.count &&
                      (sn1 == 0 || memcmp(sids1, r1.ids,
                                          sn1 * sizeof(uint32_t)) == 0),
                  "t5 sys 1t ids equal the seq reference");
            tsimd_token_sequence_free(&r1);
            tsimd_sys_free(sids1);
            uint32_t *sids4 = NULL;
            size_t sn4 = 0;
            check(tsimd_sys_encode(h, big, big_len, 4, &sids4, &sn4) == 0,
                  "t5 sys encode 4t past the into cap");
            tsimd_token_sequence_t r4;
            check(tsimd_tokenizer_encode_parallel(&ref, big, big_len, 4,
                                                  &r4) == 0,
                  "t5 parallel reference 4t");
            check(sn4 == r4.count &&
                      (sn4 == 0 || memcmp(sids4, r4.ids,
                                          sn4 * sizeof(uint32_t)) == 0),
                  "t5 sys 4t ids equal the parallel reference");
            tsimd_token_sequence_free(&r4);
            tsimd_sys_free(sids4);
            uint32_t *dids = NULL;
            size_t dn = 0;
            check(tsimd_sys_encode(h, big, big_len, 4, &dids, &dn) == 0,
                  "t5 sys encode for the decode check");
            tsimd_byte_sequence_t d1;
            memset(&d1, 0, sizeof d1);
            check(tsimd_tokenizer_decode(&ref, dids, dn, 0, &d1) == 0,
                  "simple decode of the 16MiB t5 ids");
            size_t need = 0;
            check(tsimd_tokenizer_decode_parallel(&ref, dids, dn, 0, NULL, 0,
                                                  &need, 0) == 0,
                  "decode_parallel count mode");
            uint8_t *dref = malloc(need);
            check(dref != NULL, "alloc decode reference buffer");
            size_t got2 = 0;
            check(tsimd_tokenizer_decode_parallel(&ref, dids, dn, 0, dref,
                                                  need, &got2, 0) == 0,
                  "decode_parallel write mode");
            check(d1.len == got2 && got2 > 0 &&
                      memcmp(d1.bytes, dref, got2) == 0,
                  "simple decode equals the parallel decode");
            free(dref);
            tsimd_byte_sequence_free(&d1);
            tsimd_sys_free(dids);
            free(big);
            tsimd_tokenizer_free(&ref);
        }
        tsimd_sys_close(h);
    }
    h = tsimd_sys_open(QWEN, "json", "auto", err, sizeof err);
    check(h != NULL, "open qwen2.5");
    if (h) tsimd_sys_close(h);

    h = tsimd_sys_open("/nonexistent/model.json", "json", "auto", err,
                       sizeof err);
    check(h == NULL && err[0] != 0, "a missing model must fail with a message");
    h = tsimd_sys_open(CL100K, "nonsense", "auto", err, sizeof err);
    check(h == NULL && err[0] != 0, "a bad format must fail with a message");
    h = tsimd_sys_open(CL100K, "tiktoken", "nonsense", err, sizeof err);
    check(h == NULL && err[0] != 0, "a bad scheme must fail with a message");
    h = tsimd_sys_open(NULL, "json", "auto", err, sizeof err);
    check(h == NULL, "a NULL path must fail");
    tsimd_sys_close(NULL);

    if (fails) {
        fprintf(stderr, "%d sys API checks FAILED\n", fails);
        return 1;
    }
    printf("test_sys: the opaque C ABI loads, encodes, decodes and reports "
           "errors\n");
    return 0;
}
