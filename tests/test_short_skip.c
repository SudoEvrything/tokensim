#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define REPO TSIMD_SRC_DIR

typedef struct {
    const char *name;
    tsimd_model_format_t fmt;
    const char *p1;
    const char *p2;
} fam_t;

static const fam_t k_fams[] = {
    {"o200k", TSIMD_MODEL_TIKTOKEN, REPO "/models/o200k_base.tiktoken", NULL},
    {"cl100k", TSIMD_MODEL_TIKTOKEN, REPO "/models/cl100k_base.tiktoken", NULL},
    {"deepseek", TSIMD_MODEL_TOKENIZER_JSON,
     REPO "/models/reference/deepseek_tokenizer.json", NULL},
    {"glm4", TSIMD_MODEL_TOKENIZER_JSON,
     REPO "/models/reference/glm4_tokenizer.json", NULL},
    {"mistral", TSIMD_MODEL_SENTENCEPIECE,
     REPO "/models/reference/mistral7b_tokenizer.model", NULL},
};

static unsigned long long hash_ids(const uint32_t *ids, size_t n) {
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= ids[i];
        h *= 1099511628211ull;
    }
    return h;
}

static int encode_hash(const fam_t *fm, const uint8_t *data, size_t len,
                       unsigned long long *h, size_t *n_out) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (strcmp(fm->name, "o200k") == 0) sch = TSIMD_PRETOK_O200K;
    else if (strcmp(fm->name, "cl100k") == 0) sch = TSIMD_PRETOK_CL100K;
    if (tsimd_tokenizer_load_model(&t, fm->fmt, fm->p1, fm->p2, &sch, NULL,
                                  0) != 0)
        return -1;
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(&t, data, len, &seq) != 0) {
        tsimd_tokenizer_free(&t);
        return -1;
    }
    *h = hash_ids(seq.ids, seq.count);
    *n_out = seq.count;
    tsimd_token_sequence_free(&seq);
    tsimd_tokenizer_free(&t);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--child") == 0) {
        setenv("TSIMD_DISABLE_PCACHE", "1", 1);
        int which = atoi(argv[2]);
        FILE *f = fopen(REPO "/testdata/document.txt", "rb");
        if (!f) return 1;
        static uint8_t buf[1 << 20];
        size_t n = fread(buf, 1, sizeof buf, f);
        fclose(f);
        unsigned long long h = 0;
        size_t cnt = 0;
        if (encode_hash(&k_fams[which], buf, n / 8, &h, &cnt) != 0) return 2;
        printf("%llu %zu\n", h, cnt);
        return 0;
    }

    FILE *f = fopen(REPO "/testdata/document.txt", "rb");
    if (!f) {
        fprintf(stderr, "FAIL: corpus\n");
        return 1;
    }
    static uint8_t buf[1 << 20];
    size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    size_t len = n / 8;

    int fail = 0, checks = 0;
    for (size_t i = 0; i < sizeof(k_fams) / sizeof(k_fams[0]); i++) {
        unsigned long long ha = 0, hb = 0;
        size_t na = 0, nb = 0;
        if (encode_hash(&k_fams[i], buf, len, &ha, &na) != 0) {
            fprintf(stderr, "FAIL %s: warm encode\n", k_fams[i].name);
            fail = 1;
            continue;
        }
        int pipefd[2];
        if (pipe(pipefd) != 0) return 1;
        pid_t pid = fork();
        if (pid == 0) {
            close(pipefd[0]);
            dup2(pipefd[1], STDOUT_FILENO);
            close(pipefd[1]);
            char idx[8];
            snprintf(idx, sizeof idx, "%zu", i);
            execl(argv[0], argv[0], "--child", idx, (char *)NULL);
            _exit(3);
        }
        close(pipefd[1]);
        char out[128];
        ssize_t got = read(pipefd[0], out, sizeof out - 1);
        close(pipefd[0]);
        int status = 0;
        waitpid(pid, &status, 0);
        if (got <= 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "FAIL %s: child failed\n", k_fams[i].name);
            fail = 1;
            continue;
        }
        out[got] = 0;
        sscanf(out, "%llu %zu", &hb, &nb);
        checks++;
        if (ha != hb || na != nb) {
            fprintf(stderr,
                    "FAIL %s: cache-on %llu/%zu vs cache-off %llu/%zu\n",
                    k_fams[i].name, ha, na, hb, nb);
            fail = 1;
        } else {
            printf("%-10s %zu ids, hash %llu equal both ways\n", k_fams[i].name,
                   na, ha);
        }
    }
    printf("test_short_skip: %d checks, %s\n", checks, fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}
