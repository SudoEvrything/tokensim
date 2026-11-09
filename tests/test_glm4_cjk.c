#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;

static void check(int cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

static void encode_expect(tsimd_tokenizer_t *t, const uint8_t *data,
                          size_t len, const uint32_t *want, size_t want_n,
                          const char *what) {
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    check(tsimd_tokenizer_encode(t, data, len, &seq) == 0, what);
    check(seq.count == want_n, what);
    if (seq.count == want_n)
        check(memcmp(seq.ids, want, want_n * sizeof(uint32_t)) == 0, what);
    tsimd_byte_sequence_t back;
    memset(&back, 0, sizeof back);
    check(tsimd_tokenizer_decode(t, seq.ids, seq.count, 0, &back) == 0,
          "decode");
    check(back.len == len && memcmp(back.bytes, data, len) == 0,
          "round trip");
    tsimd_byte_sequence_free(&back);
    tsimd_token_sequence_free(&seq);
}

int main(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GEM4;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON,
                                   TSIMD_SRC_DIR
                                   "/models/reference/glm4_tokenizer.json",
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "SKIP: no glm4 model\n");
        return 0;
    }
    uint8_t in1[68];
    in1[0] = ' ';
    in1[1] = 0xe5;
    memset(in1 + 2, 0xa5, 66);
    uint32_t want1[65];
    want1[0] = 112825;
    for (int i = 1; i < 65; i++) want1[i] = 98;
    encode_expect(&t, in1, sizeof in1, want1, 65,
                  "space + lead + continuation run");

    uint8_t in2[13];
    in2[0] = ' ';
    for (int i = 0; i < 4; i++) {
        in2[1 + i * 3] = 0xe5;
        in2[2 + i * 3] = 0xa5;
        in2[3 + i * 3] = 0xa5;
    }
    uint32_t want2[6] = {4891, 98, 98, 99767, 99767, 99767};
    encode_expect(&t, in2, sizeof in2, want2, 6,
                  "space + four CJK chars");

    uint8_t in3[11];
    in3[0] = ' ';
    for (int i = 0; i < 5; i++) {
        in3[1 + i * 2] = 0xc3;
        in3[2 + i * 2] = 0xa5;
    }
    uint32_t want3[5] = {13068, 3785, 3785, 3785, 3785};
    encode_expect(&t, in3, sizeof in3, want3, 5,
                  "space + five latin-1 letters");

    tsimd_token_sequence_t s1;
    memset(&s1, 0, sizeof s1);
    tsimd_token_sequence_t s6;
    memset(&s6, 0, sizeof s6);
    size_t big_len = (1u << 18) + 1;
    uint8_t *big = malloc(big_len);
    check(big != NULL, "big alloc");
    if (big) {
        big[0] = ' ';
        big[1] = 0xe5;
        memset(big + 2, 0xa5, big_len - 2);
        check(tsimd_tokenizer_encode(&t, big, big_len, &s1) == 0, "big 1t");
        check(tsimd_tokenizer_encode_parallel(&t, big, big_len, 6, &s6) == 0,
              "big 6t");
        check(s1.count == s6.count &&
                  memcmp(s1.ids, s6.ids, s1.count * sizeof(uint32_t)) == 0,
              "big parallel equals sequential");
        tsimd_byte_sequence_t back;
        memset(&back, 0, sizeof back);
        check(tsimd_tokenizer_decode(&t, s1.ids, s1.count, 0, &back) == 0,
              "big decode");
        check(back.len == big_len && memcmp(back.bytes, big, big_len) == 0,
              "big round trip");
        tsimd_byte_sequence_free(&back);
        free(big);
    }
    tsimd_token_sequence_free(&s1);
    tsimd_token_sequence_free(&s6);
    tsimd_tokenizer_free(&t);
    if (fails) {
        fprintf(stderr, "%d glm4 cjk checks FAILED\n", fails);
        return 1;
    }
    printf("test_glm4_cjk: the continuation-run spans and the "
           "ignore_merges pins hold\n");
    return 0;
}
