#include "tokenizer.h"
#include "input.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *EXTRAS[] = {"ab", "abc", "abcd", " x"};
#define VOCAB_SIZE (256 + 4)
#define SPECIAL_EOT VOCAB_SIZE

static int write_vocab_file(const char *path) {
    size_t len;
    char *buf = tsimd_test_make_vocab(EXTRAS, 4, &len);
    if (!buf) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) { free(buf); return -1; }
    size_t wrote = fwrite(buf, 1, len, f);
    fclose(f);
    free(buf);
    return wrote == len ? 0 : -1;
}

static int check_encode(tsimd_tokenizer_t *t, const char *text,
                        const uint32_t *want, size_t n) {
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, (const uint8_t *)text, strlen(text),
                              &seq) != 0)
        return 1;
    int fail = 0;
    if (seq.count != n) {
        fprintf(stderr, "\"%s\": got %zu tokens, want %zu\n", text, seq.count, n);
        fail = 1;
    } else {
        for (size_t i = 0; i < n; i++) {
            if (seq.ids[i] != want[i]) {
                fprintf(stderr, "\"%s\" token %zu: got %u, want %u\n", text, i,
                        seq.ids[i], want[i]);
                fail = 1;
            }
        }
    }
    tsimd_token_sequence_free(&seq);
    return fail;
}

static int run_golden_tokens(const char *dir, const char *name,
                             const char *scheme, const char *vocab_env) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    size_t flen;
    char *file = tsimd_test_read_file(path, &flen);
    if (!file) {
        printf("(fixture %s not present, skipping)\n", name);
        return 0;
    }
    const char *vocab_path = getenv(vocab_env);
    if (!vocab_path) {
        printf("(fixture %s present but %s unset, skipping)\n", name, vocab_env);
        free(file);
        return 0;
    }
    tsimd_tokenizer_t t;
    if (tsimd_tokenizer_load(&t, vocab_path,
                            strcmp(scheme, "cl100k") == 0
                                ? TSIMD_PRETOK_CL100K : TSIMD_PRETOK_GPT2,
                            NULL, 0) != 0) {
        fprintf(stderr, "failed to load vocab %s\n", vocab_path);
        free(file);
        return 1;
    }
    int failures = 0, cases = 0;
    char *line = file;
    while (line < file + flen) {
        char *nl = strchr(line, '\n');
        if (!nl) break;
        *nl = 0;
        char *tab = strchr(line, '\t');
        if (!tab) { line = nl + 1; continue; }
        *tab = 0;
        uint8_t input[4096];
        size_t ilen;
        if (tsimd_test_unescape(line, strlen(line), input, sizeof(input),
                               &ilen) != 0) {
            failures++;
            line = nl + 1;
            continue;
        }
        uint32_t want[8192];
        size_t n = 0;
        char *p = tab + 1;
        while (*p && n < 8192) {
            char *comma = strchr(p, ',');
            if (comma) *comma = 0;
            if (*p) want[n++] = (uint32_t)strtoul(p, NULL, 10);
            if (!comma) break;
            p = comma + 1;
        }
        cases++;
        tsimd_token_sequence_t seq;
        if (tsimd_tokenizer_encode(&t, input, ilen, &seq) != 0 ||
            seq.count != n ||
            (n > 0 && memcmp(seq.ids, want, n * sizeof(uint32_t)) != 0)) {
            fprintf(stderr, "golden mismatch on: %.*s (got %zu ids, want %zu)\n",
                    (int)(ilen > 80 ? 80 : ilen), (const char *)input,
                    seq.count, n);
            failures++;
        }
        tsimd_token_sequence_free(&seq);
        line = nl + 1;
    }
    tsimd_tokenizer_free(&t);
    free(file);
    printf("golden tokens %s: %d cases, %d failures\n", name, cases, failures);
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *fixtures = argc > 1 ? argv[1] : "tests/fixtures";
    char vocab_path_tmpl[] = "/tmp/tsimd_test_vocab_XXXXXX";
    int vocab_fd = mkstemp(vocab_path_tmpl);
    if (vocab_fd < 0) return 1;
    close(vocab_fd);
    const char *vocab_path = vocab_path_tmpl;

    TEST_ASSERT(write_vocab_file(vocab_path) == 0);

    {
        tsimd_input_t in;
        TEST_ASSERT(tsimd_input_read_file(vocab_path, &in) == 0);
        size_t len;
        char *want = tsimd_test_make_vocab(EXTRAS, 4, &len);
        TEST_ASSERT(in.length == len);
        TEST_ASSERT(memcmp(in.data, want, len) == 0);
        free(want);
        tsimd_input_release(&in);
        TEST_ASSERT(tsimd_input_read_file("/nonexistent/zzz", &in) == -1);
    }

    tsimd_tokenizer_t t;
    const tsimd_added_token_t specials[] = {
        {"<|eot|>", 0, 7, 1},
    };
    TEST_ASSERT(tsimd_tokenizer_load(&t, vocab_path, TSIMD_PRETOK_GPT2,
                                    specials, 1) == 0);
    TEST_ASSERT_EQ(t.vocab.vocab_size, VOCAB_SIZE);

    {
        static const struct { const char *s; uint32_t id; } seeded[] = {
            {"ab", 256}, {"abc", 257}, {"abcd", 258}, {" x", 259},
        };
        for (size_t i = 0; i < sizeof(seeded) / sizeof(seeded[0]); i++) {
            uint32_t n_ids = 0;
            uint32_t *r = tsimd_pcache_lookup(
                &t.cache, (const uint8_t *)seeded[i].s,
                (uint32_t)strlen(seeded[i].s), &n_ids);
            TEST_ASSERT(r != NULL);
            TEST_ASSERT_EQ(n_ids, 1);
            TEST_ASSERT_EQ(r[0], seeded[i].id);
        }

        uint32_t n_ids = 99;
        TEST_ASSERT(tsimd_pcache_lookup(&t.cache, (const uint8_t *)"zz",
                                        2, &n_ids) == NULL);
    }

    {
        const uint32_t want[] = {258, SPECIAL_EOT, 259, 256};
        TEST_ASSERT(check_encode(&t, "abcd<|eot|> xab", want, 4) == 0);
    }
    {
        const uint32_t want[] = {256, SPECIAL_EOT, SPECIAL_EOT, 256};
        TEST_ASSERT(check_encode(&t, "ab<|eot|><|eot|>ab", want, 4) == 0);
    }

    {
        const uint32_t want[] = {256, SPECIAL_EOT, 'c', 'd'};
        TEST_ASSERT(check_encode(&t, "ab<|eot|>cd", want, 4) == 0);
    }

    {
        const uint32_t want[] = {'a', ' ', 'b', ' ', 'c'};
        TEST_ASSERT(check_encode(&t, "a b c", want, 5) == 0);
    }

    {
        tsimd_token_sequence_t seq;
        TEST_ASSERT(tsimd_tokenizer_encode(&t, (const uint8_t *)"", 0, &seq) == 0);
        TEST_ASSERT_EQ(seq.count, 0);
        tsimd_token_sequence_free(&seq);
    }

    {
        const uint8_t junk[] = {0x00, 0xC3, 0x28, 0xFF, 0xF0, 0x9F, 0x9A, 0x80};
        tsimd_token_sequence_t seq;
        TEST_ASSERT(tsimd_tokenizer_encode(&t, junk, sizeof(junk), &seq) == 0);
        TEST_ASSERT(seq.count >= 4);
        tsimd_token_sequence_free(&seq);
    }

    tsimd_tokenizer_free(&t);
    remove(vocab_path);

    TEST_ASSERT(run_golden_tokens(fixtures, "golden_tokens_gpt2.tsv",
                                  "gpt2", "TSIMD_GPT2_VOCAB") == 0);
    TEST_ASSERT(run_golden_tokens(fixtures, "golden_tokens_cl100k.tsv",
                                  "cl100k", "TSIMD_CL100K_VOCAB") == 0);

    printf("test_tokenizer: OK\n");
    return 0;
}
