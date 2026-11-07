#include "tokenizer.h"
#include "bpe.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPO TSIMD_SRC_DIR

static int g_fail;
static int checks;

static const uint8_t k_sep[3] = {0xE2, 0x96, 0x81};

static void chk(int ok, const char *what) {
    checks++;
    if (!ok) {
        fprintf(stderr, "FAIL %s\n", what);
        g_fail = 1;
    }
}

static uint8_t *to_metaspace(const uint8_t *in, size_t n, size_t *out_n) {
    size_t cap = n * 3 + 64;
    uint8_t *b = malloc(cap);
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        if (in[i] == ' ' || in[i] == '\n') {
            memcpy(b + o, k_sep, 3);
            o += 3;
        } else {
            b[o++] = in[i];
        }
    }
    *out_n = o;
    return b;
}

static int merge_ids(const tsimd_vocab_t *v, const uint8_t *bytes, size_t n,
                     size_t target, uint32_t **ids, size_t *count) {
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, v->max_token_len) != 0) return -1;
    if (tsimd_bpe_encode_spm_blocked(v, &ws, bytes, (uint32_t)n, target) != 0) {
        tsimd_bpe_workspace_free(&ws);
        return -1;
    }
    *ids = malloc((ws.out_count ? ws.out_count : 1) * sizeof(uint32_t));
    memcpy(*ids, ws.out, ws.out_count * sizeof(uint32_t));
    *count = ws.out_count;
    tsimd_bpe_workspace_free(&ws);
    return 0;
}

static int run_model(const char *name, tsimd_model_format_t fmt, const char *path,
                     const uint8_t *ms, size_t mslen) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, fmt, path, NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "FAIL %s: load\n", name);
        g_fail = 1;
        checks++;
        return -1;
    }
    int coarse = tsimd_vocab_spm_seam_ok(&t.vocab);
    checks++;
    printf("%s: coarse seam = %d, vocab %u\n", name, coarse, t.vocab.vocab_size);

    uint32_t *ref = NULL;
    size_t nref = 0;
    if (merge_ids(&t.vocab, ms, mslen, 0, &ref, &nref) != 0) {
        fprintf(stderr, "FAIL %s: plain merge\n", name);
        g_fail = 1;
        checks++;
        tsimd_tokenizer_free(&t);
        return -1;
    }
    chk(nref > 0, "plain merge produced ids");

    static const size_t targets[] = {1, 2, 4, 8, 16, 64, 1024, 16384};
    for (size_t k = 0; k < sizeof(targets) / sizeof(targets[0]); k++) {
        uint32_t *ids = NULL;
        size_t n = 0;
        if (merge_ids(&t.vocab, ms, mslen, targets[k], &ids, &n) != 0) {
            fprintf(stderr, "FAIL %s: split merge target=%zu\n", name,
                    targets[k]);
            g_fail = 1;
            checks++;
            continue;
        }
        checks++;
        if (n != nref || memcmp(ids, ref, nref * sizeof(uint32_t)) != 0) {
            size_t i = 0;
            while (i < n && i < nref && ids[i] == ref[i]) i++;
            fprintf(stderr,
                    "FAIL %s: target=%zu split output differs (%zu vs %zu ids,"
                    " first diff %zu)\n",
                    name, targets[k], n, nref, i);
            g_fail = 1;
        }
        free(ids);
    }

    {
        size_t pos = 0;
        size_t seams = 0, bad = 0;
        for (int k = 0; k < 200 && pos + 4 < mslen; k++) {
            size_t s = tsimd_spm_seam_find(&t.vocab, ms, mslen, pos + 8,
                                           pos + 4);
            if (s == 0) break;
            if (memcmp(ms + s, k_sep, 3) != 0) bad++;
            seams++;
            pos = s;
        }
        chk(bad == 0, "every seam sits on a marker");
        printf("%s: %zu word seams sampled over %zu bytes, %d off-marker\n",
               name, seams, mslen, (int)bad);
    }

    free(ref);
    tsimd_tokenizer_free(&t);
    return 0;
}

int main(void) {
    FILE *f = fopen(REPO "/testdata/document.txt", "rb");
    if (!f) {
        fprintf(stderr, "FAIL: corpus\n");
        return 1;
    }
    static uint8_t buf[1 << 19];
    size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    size_t mslen = 0;
    uint8_t *ms = to_metaspace(buf, n, &mslen);
    if (!ms) return 1;

    run_model("gemma4", TSIMD_MODEL_TOKENIZER_JSON,
              REPO "/models/reference/gemma4_tokenizer.json", ms, mslen);
    run_model("mistral7b", TSIMD_MODEL_SENTENCEPIECE,
              REPO "/models/reference/mistral7b_tokenizer.model", ms, mslen);
    free(ms);
    printf("test_spm_word_seam: %d checks, %s\n", checks, g_fail ? "FAIL" : "OK");
    return g_fail ? 1 : 0;
}
