#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;

static void write_file(const char *path, const char *content) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fails++;
        fprintf(stderr, "FAIL cannot write %s\n", path);
        return;
    }
    fputs(content, f);
    fclose(f);
}

static void b64_line(FILE *f, const uint8_t *b, size_t n, uint32_t rank) {
    static const char alpha[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    fprintf(f, "%c%c%c%c ", alpha[b[0] >> 2],
            alpha[((b[0] & 3) << 4) | (n > 1 ? (b[1] >> 4) : 0)],
            n > 1 ? alpha[((b[1] & 15) << 2) | (n > 2 ? (b[2] >> 6) : 0)] : '=',
            n > 2 ? alpha[b[2] & 63] : '=');
    fprintf(f, "%u\n", rank);
}

static int load_scheme(const char *path, tsimd_pretok_scheme_t sch) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t det = sch;
    int rc = tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TIKTOKEN, path, NULL,
                                        &det, NULL, 0);
    if (rc == 0) tsimd_tokenizer_free(&t);
    return rc;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "/tmp";
    char raw[512];
    snprintf(raw, sizeof raw, "%s/tsimd_guard_raw.tiktoken", dir);

    FILE *fr = fopen(raw, "wb");
    if (!fr) {
        fails++;
        fprintf(stderr, "FAIL cannot write fixture\n");
    }
    uint32_t rank = 0;
    for (unsigned b = 0; b <= 0xFF && !fails; b++) {
        uint8_t one = (uint8_t)b;
        b64_line(fr, &one, 1, rank++);
    }
    fclose(fr);

    if (load_scheme(raw, TSIMD_PRETOK_GPT2) != 0) {
        fails++;
        fprintf(stderr, "FAIL small raw-byte vocab rejected scheme gpt2\n");
    }
    if (load_scheme(raw, TSIMD_PRETOK_GPT2D) != 0) {
        fails++;
        fprintf(stderr, "FAIL small raw-byte vocab rejected scheme gpt2d\n");
    }
    if (load_scheme(raw, TSIMD_PRETOK_CL100K) != 0) {
        fails++;
        fprintf(stderr, "FAIL raw-byte vocab rejected scheme cl100k\n");
    }
    if (load_scheme(raw, TSIMD_PRETOK_O200K) != 0) {
        fails++;
        fprintf(stderr, "FAIL raw-byte vocab rejected scheme o200k\n");
    }
    if (load_scheme(raw, TSIMD_PRETOK_QWEN25) != 0) {
        fails++;
        fprintf(stderr, "FAIL raw-byte vocab rejected scheme qwen25\n");
    }

    if (argc > 2) {
        if (load_scheme(argv[2], TSIMD_PRETOK_GPT2) == 0) {
            fails++;
            fprintf(stderr, "FAIL %s accepted scheme gpt2\n", argv[2]);
        }
        if (load_scheme(argv[2], TSIMD_PRETOK_GPT2D) == 0) {
            fails++;
            fprintf(stderr, "FAIL %s accepted scheme gpt2d\n", argv[2]);
        }
        if (load_scheme(argv[2], TSIMD_PRETOK_O200K) != 0) {
            fails++;
            fprintf(stderr, "FAIL %s rejected scheme o200k\n", argv[2]);
        }
        if (load_scheme(argv[2], TSIMD_PRETOK_CL100K) != 0) {
            fails++;
            fprintf(stderr, "FAIL %s rejected scheme cl100k\n", argv[2]);
        }
    }

    if (fails == 0) printf("test_scheme_guard: OK\n");
    return fails ? 1 : 0;
}
