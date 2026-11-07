#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const char *g_fixtures_dir = "tests/fixtures";
static char fx[2][512];
static int fx_slot;

static const char *fixture(const char *name) {
    fx_slot = (fx_slot + 1) % 2;
    snprintf(fx[fx_slot], sizeof(fx[0]), "%s/%s", g_fixtures_dir, name);
    return fx[fx_slot];
}

static char g_scratch[256];
static const char *wfixture(const char *name) {
    if (!g_scratch[0]) {
        snprintf(g_scratch, sizeof(g_scratch), "/tmp/tsimd_fx_XXXXXX");
        if (!mkdtemp(g_scratch)) {
            perror("mkdtemp");
            g_scratch[0] = '.';
            g_scratch[1] = '\0';
        }
    }
    fx_slot = (fx_slot + 1) % 2;
    snprintf(fx[fx_slot], sizeof(fx[0]), "%s/%s", g_scratch, name);
    return fx[fx_slot];
}

static int enc(tsimd_tokenizer_t *t, const char *text, uint32_t *ids,
               size_t *count) {
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, (const uint8_t *)text, strlen(text),
                              &seq) != 0)
        return -1;
    if (seq.count > 16) {
        tsimd_token_sequence_free(&seq);
        return -1;
    }
    memcpy(ids, seq.ids, seq.count * sizeof(uint32_t));
    *count = seq.count;
    tsimd_token_sequence_free(&seq);
    return 0;
}

static int check(tsimd_tokenizer_t *t, const char *text,
                 const uint32_t *want, size_t n) {
    uint32_t got[16];
    size_t got_n = 0;
    if (enc(t, text, got, &got_n) != 0) return 1;
    if (got_n != n || memcmp(got, want, n * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "\"%s\": got %zu ids, want %zu\n", text, got_n, n);
        for (size_t i = 0; i < got_n; i++)
            fprintf(stderr, "  got[%zu]=%u\n", i, got[i]);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1) g_fixtures_dir = argv[1];

    tsimd_tokenizer_t tiktok, gpt2t, tjt;
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    tsimd_added_token_t eot = {"<|eot|>", 0, 7, 1};

    TEST_ASSERT(tsimd_tokenizer_load(&tiktok, fixture("mini.tiktoken"),
                                    TSIMD_PRETOK_GPT2, &eot, 1) == 0);
    TEST_ASSERT(tsimd_tokenizer_load_model(&gpt2t, TSIMD_MODEL_GPT2,
                                          fixture("mini_encoder.json"),
                                          fixture("mini_vocab.bpe"), &scheme,
                                          &eot, 1) == 0);
    scheme = TSIMD_PRETOK_GPT2;
    TEST_ASSERT(tsimd_tokenizer_load_model(&tjt, TSIMD_MODEL_TOKENIZER_JSON,
                                          fixture("mini_tokenizer.json"),
                                          NULL, &scheme, &eot, 1) == 0);
    TEST_ASSERT_EQ(scheme, TSIMD_PRETOK_GPT2);
    TEST_ASSERT_EQ(tiktok.vocab.vocab_size, 260);
    TEST_ASSERT_EQ(gpt2t.vocab.vocab_size, 261);
    TEST_ASSERT_EQ(tjt.vocab.vocab_size, 260);

    TEST_ASSERT_EQ(tjt.added_count, 3);
    TEST_ASSERT(strcmp(tjt.added[0].text, "<|eot|>") == 0);
    TEST_ASSERT_EQ(tjt.added[0].id, 260);
    TEST_ASSERT_EQ(tjt.added[0].is_special, 1);
    TEST_ASSERT(strcmp(tjt.added[1].text, "hello") == 0);
    TEST_ASSERT_EQ(tjt.added[1].id, 261);
    TEST_ASSERT_EQ(tjt.added[1].is_special, 0);

    TEST_ASSERT(strcmp(tjt.added[2].text, "<|eot|>") == 0);
    TEST_ASSERT_EQ(tjt.added[2].id, 262);

    {

        const uint32_t w1[] = {257, 257};

        const uint32_t w2[] = {258};

        const uint32_t w3[] = {259};

        const uint32_t w4[] = {256, 259};

        const uint32_t w4b[] = {97, 32, 259};

        const uint32_t w5[] = {'q', 'w', 'e'};
        TEST_ASSERT(check(&tiktok, "abcabc", w1, 2) == 0);
        TEST_ASSERT(check(&gpt2t, "abcabc", w1, 2) == 0);
        TEST_ASSERT(check(&tjt, "abcabc", w1, 2) == 0);
        TEST_ASSERT(check(&tiktok, " x", w2, 1) == 0);
        TEST_ASSERT(check(&gpt2t, " x", w2, 1) == 0);
        TEST_ASSERT(check(&tjt, " x", w2, 1) == 0);
        TEST_ASSERT(check(&tiktok, "th", w3, 1) == 0);
        TEST_ASSERT(check(&gpt2t, "th", w3, 1) == 0);
        TEST_ASSERT(check(&tjt, "th", w3, 1) == 0);
        TEST_ASSERT(check(&tiktok, "abth", w4, 2) == 0);
        TEST_ASSERT(check(&gpt2t, "abth", w4, 2) == 0);
        TEST_ASSERT(check(&tjt, "abth", w4, 2) == 0);
        TEST_ASSERT(check(&tiktok, "a th", w4b, 3) == 0);
        TEST_ASSERT(check(&gpt2t, "a th", w4b, 3) == 0);
        TEST_ASSERT(check(&tjt, "a th", w4b, 3) == 0);
        TEST_ASSERT(check(&tiktok, "qwe", w5, 3) == 0);
        TEST_ASSERT(check(&gpt2t, "qwe", w5, 3) == 0);
        TEST_ASSERT(check(&tjt, "qwe", w5, 3) == 0);
    }

    {
        const uint32_t w_tik[] = {257, 260, 256};
        TEST_ASSERT(check(&tiktok, "abc<|eot|>ab", w_tik, 3) == 0);
        const uint32_t w_gpt2[] = {257, 261, 256};
        TEST_ASSERT(check(&gpt2t, "abc<|eot|>ab", w_gpt2, 3) == 0);
        const uint32_t w_tj[] = {257, 260, 256};
        TEST_ASSERT(check(&tjt, "abc<|eot|>ab", w_tj, 3) == 0);
    }

    {
        const uint32_t w[] = {'a', 261, 'b'};
        TEST_ASSERT(check(&tjt, "ahellob", w, 3) == 0);
    }

    {
        tsimd_tokenizer_t t2;
        tsimd_added_token_t e2 = {"<|x|>", 999, 5, 1};
        TEST_ASSERT(tsimd_tokenizer_load(&t2, fixture("mini.tiktoken"),
                                        TSIMD_PRETOK_GPT2, &e2, 1) == 0);
        TEST_ASSERT_EQ(t2.added[0].id, 999);
        const uint32_t w[] = {999};
        TEST_ASSERT(check(&t2, "<|x|>", w, 1) == 0);
        tsimd_tokenizer_free(&t2);
    }

    {
        tsimd_tokenizer_t t2;
        tsimd_pretok_scheme_t s2 = TSIMD_PRETOK_GPT2;

        TEST_ASSERT(tsimd_tokenizer_load_model(&t2, TSIMD_MODEL_GPT2,
                                              fixture("mini_encoder.json"),
                                              fixture("does_not_exist.bpe"),
                                              &s2, NULL, 0) != 0);

        static const char *bad_pre =
            "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
            "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pattern\":"
            "[{\"Regex\":\"[^\\\\r\\\\n\\p{Lu}]?\"}]}}";
        static const char *bad_type =
            "{\"model\":{\"type\":\"Unigram\"}}";
        static const char *truncated = "{\"model\": {\"type\": \"BPE\"";
        const char *names[] = {"bad_pre.json", "bad_type.json",
                               "truncated.json"};
        const char *docs[] = {bad_pre, bad_type, truncated};
        for (size_t i = 0; i < 3; i++) {
            FILE *f = fopen(wfixture(names[i]), "wb");
            TEST_ASSERT(f != NULL);
            fwrite(docs[i], 1, strlen(docs[i]), f);
            fclose(f);
            s2 = TSIMD_PRETOK_GPT2;
            TEST_ASSERT(tsimd_tokenizer_load_model(&t2,
                                                  TSIMD_MODEL_TOKENIZER_JSON,
                                                  wfixture(names[i]), NULL,
                                                  &s2, NULL, 0) != 0);
            remove(wfixture(names[i]));
        }

        static const char *cl100k_pat =
            "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
            "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pattern\":"
            "[{\"Regex\":\"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+\"}]}}";
        static const char *gpt2_pat =
            "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
            "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pattern\":"
            "[{\"Regex\":\"'s|'t|'re|'ve|'m|'ll|'d| ?\\\\p{L}+| ?\\\\p{N}+| ?[^\\\\s\\\\p{L}\\\\p{N}]+|\\\\s+(?!\\\\S)|\\\\s+\"}]}}";

        static const char *qwen25_regex_json =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+";
        char qwen25_doc[1100];
        snprintf(qwen25_doc, sizeof(qwen25_doc),
                 "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
                 "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pattern\":["
                 "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"%s\",\"behavior\":\"Isolated\"}},"
                 "{\"type\":\"ByteLevel\",\"use_regex\":false}]}}",
                 qwen25_regex_json);
        const char *qwen25_names[] = {"det_qwen25.json"};
        const char *qwen25_docs[] = {qwen25_doc};
        const int qwen25_want[] = {TSIMD_PRETOK_QWEN25};
        for (size_t i = 0; i < 1; i++) {
            const char *qp = wfixture(qwen25_names[i]);
            FILE *f = fopen(qp, "wb");
            TEST_ASSERT(f != NULL);
            fwrite(qwen25_docs[i], 1, strlen(qwen25_docs[i]), f);
            fclose(f);
            s2 = TSIMD_PRETOK_GPT2;

            TEST_ASSERT(tsimd_tokenizer_load_model(&t2,
                                                  TSIMD_MODEL_TOKENIZER_JSON,
                                                  qp, NULL,
                                                  &s2, NULL, 0) == 0);
            TEST_ASSERT_EQ(s2, qwen25_want[i]);
            remove(wfixture(qwen25_names[i]));
        }

        static const char *cl100k_regex_json =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+";
        char qwen_doc[1024];
        snprintf(qwen_doc, sizeof(qwen_doc),
                 "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]},"
                 "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pattern\":["
                 "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"%s\",\"behavior\":\"Isolated\"}},"
                 "{\"type\":\"ByteLevel\"}]}}",
                 cl100k_regex_json);
        const char *det_names[] = {"det_cl100k.json", "det_gpt2.json",
                                   "det_qwen.json"};
        const char *det_docs[] = {cl100k_pat, gpt2_pat, qwen_doc};
        const int det_want[] = {TSIMD_PRETOK_CL100K, TSIMD_PRETOK_GPT2,
                                TSIMD_PRETOK_CL100K};
        for (size_t i = 0; i < 3; i++) {
            FILE *f = fopen(wfixture(det_names[i]), "wb");
            TEST_ASSERT(f != NULL);
            fwrite(det_docs[i], 1, strlen(det_docs[i]), f);
            fclose(f);
            s2 = TSIMD_PRETOK_GPT2;

            TEST_ASSERT(tsimd_tokenizer_load_model(&t2,
                                                  TSIMD_MODEL_TOKENIZER_JSON,
                                                  wfixture(det_names[i]), NULL,
                                                  &s2, NULL, 0) == 0);
            TEST_ASSERT_EQ(s2, det_want[i]);
            remove(wfixture(det_names[i]));
        }

        FILE *f = fopen(wfixture("garbage.json"), "wb");
        TEST_ASSERT(f != NULL);
        fputs("not json at all", f);
        fclose(f);
        s2 = TSIMD_PRETOK_GPT2;
        TEST_ASSERT(tsimd_tokenizer_load_model(&t2, TSIMD_MODEL_GPT2,
                                              wfixture("garbage.json"), NULL,
                                              &s2, NULL, 0) != 0);
        remove(wfixture("garbage.json"));
    }

    tsimd_tokenizer_free(&tiktok);
    tsimd_tokenizer_free(&gpt2t);
    tsimd_tokenizer_free(&tjt);
    printf("test_model_formats: OK\n");
    return 0;
}
