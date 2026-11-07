#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPO TSIMD_SRC_DIR

static int g_fail;
static int checks;

typedef struct {
    const char *name;
    tsimd_model_format_t fmt;
    const char *path1;
    const char *path2;
    tsimd_pretok_scheme_t scheme;
} fam_t;

static const fam_t k_fams[] = {
    {"o200k", TSIMD_MODEL_TIKTOKEN, REPO "/models/o200k_base.tiktoken", NULL,
     TSIMD_PRETOK_O200K},
    {"cl100k", TSIMD_MODEL_TIKTOKEN, REPO "/models/cl100k_base.tiktoken", NULL,
     TSIMD_PRETOK_CL100K},
    {"deepseek", TSIMD_MODEL_TOKENIZER_JSON,
     REPO "/models/reference/deepseek_tokenizer.json", NULL, TSIMD_PRETOK_GPT2},
    {"qwen2.5", TSIMD_MODEL_TOKENIZER_JSON,
     REPO "/models/reference/qwen2.5-0.5b_tokenizer.json", NULL,
     TSIMD_PRETOK_GPT2},
    {"mistral", TSIMD_MODEL_SENTENCEPIECE,
     REPO "/models/reference/mistral7b_tokenizer.model", NULL,
     TSIMD_PRETOK_GPT2},
};

static uint8_t *load(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz <= 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    uint8_t *b = malloc((size_t)sz);
    if (!b) {
        fclose(f);
        return NULL;
    }
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) {
        free(b);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

static void chk(int ok, const char *what) {
    checks++;
    if (!ok) {
        fprintf(stderr, "FAIL %s\n", what);
        g_fail = 1;
    }
}

static void chk_eq(const uint32_t *a, size_t na, const uint32_t *b, size_t nb,
                   const char *what) {
    checks++;
    if (na != nb || memcmp(a, b, na * sizeof(uint32_t)) != 0) {
        size_t i = 0;
        while (i < na && i < nb && a[i] == b[i]) i++;
        fprintf(stderr, "FAIL %s: %zu vs %zu ids, first diff %zu\n", what, na,
                nb, i);
        g_fail = 1;
    }
}

static int run_family(const fam_t *fm, const uint8_t *doc, size_t dlen,
                      const uint8_t *doc2, size_t d2len) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = fm->scheme;
    if (tsimd_tokenizer_load_model(&t, fm->fmt, fm->path1, fm->path2, &sch, NULL,
                                  0) != 0) {
        fprintf(stderr, "FAIL %s: load\n", fm->name);
        g_fail = 1;
        checks++;
        return -1;
    }

    tsimd_token_sequence_t ref;
    if (tsimd_tokenizer_encode(&t, doc, dlen, &ref) != 0) {
        fprintf(stderr, "FAIL %s: plain encode\n", fm->name);
        g_fail = 1;
        checks++;
        tsimd_tokenizer_free(&t);
        return -1;
    }
    chk(ref.count > 0, "reference count > 0");
    chk(ref.cap >= ref.count, "reference cap >= count");

    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    if (tsimd_tokenizer_encode_into(&t, doc, dlen, &seq) != 0) {
        fprintf(stderr, "FAIL %s: encode_into fresh\n", fm->name);
        g_fail = 1;
        checks++;
        tsimd_token_sequence_free(&ref);
        tsimd_tokenizer_free(&t);
        return -1;
    }
    chk_eq(ref.ids, ref.count, seq.ids, seq.count, "fresh into == encode");
    chk(seq.cap >= seq.count, "fresh into cap >= count");

    uint32_t *p_after_first = seq.ids;
    size_t cap_after_first = seq.cap;
    if (tsimd_tokenizer_encode_into(&t, doc, dlen, &seq) != 0) {
        fprintf(stderr, "FAIL %s: encode_into reuse\n", fm->name);
        g_fail = 1;
        checks++;
    }
    checks++;
    if (seq.ids != p_after_first || seq.cap != cap_after_first) {
        fprintf(stderr, "FAIL %s: reuse reallocated (ptr %p -> %p)\n", fm->name,
                (void *)p_after_first, (void *)seq.ids);
        g_fail = 1;
    }
    chk_eq(ref.ids, ref.count, seq.ids, seq.count, "reuse == encode");

    tsimd_token_sequence_t ref2;
    memset(&ref2, 0, sizeof ref2);
    if (tsimd_tokenizer_encode(&t, doc2, d2len, &ref2) != 0) {
        fprintf(stderr, "FAIL %s: plain encode 2\n", fm->name);
        g_fail = 1;
        checks++;
    }
    if (tsimd_tokenizer_encode_into(&t, doc2, d2len, &seq) != 0) {
        fprintf(stderr, "FAIL %s: encode_into other input\n", fm->name);
        g_fail = 1;
        checks++;
    }
    chk_eq(ref2.ids, ref2.count, seq.ids, seq.count, "other input == encode");

    if (tsimd_tokenizer_encode_into(&t, doc, dlen, &seq) != 0) {
        fprintf(stderr, "FAIL %s: encode_into back to input 1\n", fm->name);
        g_fail = 1;
        checks++;
    }
    chk_eq(ref.ids, ref.count, seq.ids, seq.count, "back to input 1 == encode");

    {
        static const uint8_t tiny[] = "a b";
        tsimd_token_sequence_t small;
        memset(&small, 0, sizeof small);
        if (tsimd_tokenizer_encode_into(&t, tiny, sizeof tiny - 1, &small) !=
            0) {
            fprintf(stderr, "FAIL %s: seed small buffer\n", fm->name);
            g_fail = 1;
            checks++;
        }
        size_t cap_small = small.cap;
        uint32_t *p_small = small.ids;
        if (tsimd_tokenizer_encode_into(&t, doc, dlen, &small) != 0) {
            fprintf(stderr, "FAIL %s: grow from small\n", fm->name);
            g_fail = 1;
            checks++;
        }
        chk_eq(ref.ids, ref.count, small.ids, small.count,
               "grown from small == encode");
        chk(small.cap >= ref.count, "grown cap >= count");
        checks++;
        if (cap_small >= ref.count && small.ids != p_small) {
            fprintf(stderr, "FAIL %s: reallocated with room to spare\n",
                    fm->name);
            g_fail = 1;
        }
        tsimd_token_sequence_free(&small);
    }

    {
        tsimd_token_sequence_t z;
        memset(&z, 0, sizeof z);
        size_t need = tsimd_tokenizer_encode_bound(&t, dlen);
        chk(need >= ref.count, "bound >= count");
        uint32_t *buf = malloc(need * sizeof(uint32_t));
        chk(buf != NULL, "bound buffer allocated");
        if (buf) {
            memset(buf, 0xCD, need * sizeof(uint32_t));
            size_t got = 0;
            int rc = tsimd_tokenizer_encode_into_raw(&t, doc, dlen, buf, need,
                                                    &got);
            chk(rc == 0, "raw into bounded buffer");
            chk_eq(ref.ids, ref.count, buf, got, "raw == encode");
            free(buf);
        }
        (void)z;
    }

    tsimd_token_sequence_free(&seq);
    tsimd_token_sequence_free(&ref);
    tsimd_token_sequence_free(&ref2);
    tsimd_tokenizer_free(&t);
    return 0;
}

int main(void) {
    size_t dlen = 0, d2len = 0;
    uint8_t *doc = load(REPO "/testdata/document.txt", &dlen);
    uint8_t *doc2 = load(REPO "/testdata/owt.txt", &d2len);
    if (!doc || !doc2) {
        fprintf(stderr, "FAIL: cannot read corpora\n");
        return 1;
    }
    size_t shrink = dlen / 4 ? dlen / 4 : dlen;
    size_t shrink2 = d2len > 65536 ? 65536 : d2len;
    for (size_t i = 0; i < sizeof(k_fams) / sizeof(k_fams[0]); i++)
        run_family(&k_fams[i], doc, shrink, doc2, shrink2);
    free(doc);
    free(doc2);
    printf("test_encode_into: %d checks, %s\n", checks,
           g_fail ? "FAIL" : "OK");
    return g_fail ? 1 : 0;
}
