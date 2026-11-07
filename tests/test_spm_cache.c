#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define REPO TSIMD_SRC_DIR

static const char *k_models[] = {
    REPO "/models/reference/gemma4_tokenizer.json",
    REPO "/models/reference/mistral7b_tokenizer.model",
};

static unsigned long long hash_ids(const uint32_t *ids, size_t n) {
    unsigned long long h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) {
        h ^= ids[i];
        h *= 1099511628211ull;
    }
    return h;
}

static int run_one(const char *path, unsigned long long *h, size_t *n_out) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    tsimd_model_format_t fmt = TSIMD_MODEL_TOKENIZER_JSON;
    size_t pl = strlen(path);
    if (pl > 6 && strcmp(path + pl - 6, ".model") == 0)
        fmt = TSIMD_MODEL_SENTENCEPIECE;
    if (tsimd_tokenizer_load_model(&t, fmt, path, NULL, &sch, NULL, 0) != 0)
        return -1;
    FILE *f = fopen(REPO "/testdata/document.txt", "rb");
    if (!f) {
        tsimd_tokenizer_free(&t);
        return -1;
    }
    static uint8_t buf[1 << 20];
    size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    tsimd_token_sequence_t seq;
    unsigned long long acc = 1469598103934665603ull;
    size_t count = 0;
    int rc = 0;
    for (int pass = 0; pass < 3 && rc == 0; pass++) {
        size_t len = n / 8 + (size_t)pass * 4096;
        if (len > n) len = n;
        if (tsimd_tokenizer_encode(&t, buf, len, &seq) != 0) {
            rc = -1;
            break;
        }
        acc ^= hash_ids(seq.ids, seq.count);
        acc *= 1099511628211ull;
        count += seq.count;
        tsimd_token_sequence_free(&seq);
    }
    tsimd_tokenizer_free(&t);
    if (rc != 0) return -1;
    *h = acc;
    *n_out = count;
    return 0;
}

static int run_long(unsigned long long *h, size_t *n_out, size_t *hits,
                    size_t *ins) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON,
                                   k_models[0], NULL, &sch, NULL, 0) != 0)
        return -1;
    static const char *words[] = {
        "Zxcvbnmasdfghj ", "Qwertyxzplmnb ", "Kjhgfdsapoiuy ",
        "Mnbvcxzlkjhgf ", "Poiuytrewqasd ", "Lkjhgfdsamnbvc ",
        "the quick brown ", " don't stop believing ",
    };
    static uint8_t buf[1 << 20];
    size_t n = 0;
    for (int rep = 0; rep < 6000 && n + 64 < sizeof buf; rep++) {
        for (size_t w = 0; w < sizeof(words) / sizeof(words[0]); w++) {
            size_t l = strlen(words[w]);
            if (n + l + 1 > sizeof buf) break;
            memcpy(buf + n, words[w], l);
            n += l;
        }
    }
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(&t, buf, n, &seq) != 0) {
        tsimd_tokenizer_free(&t);
        return -1;
    }
    *h = hash_ids(seq.ids, seq.count);
    *n_out = seq.count;
    size_t m = 0, i2 = 0;
    tsimd_plong_stats(hits, &m, &i2);
    if (ins) *ins = i2;
    tsimd_token_sequence_free(&seq);
    tsimd_tokenizer_free(&t);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--long") == 0) {
        unsigned long long h = 0;
        size_t n = 0, hits = 0, ins = 0;
        if (run_long(&h, &n, &hits, &ins) != 0) return 2;
        if (!getenv("TSIMD_SKIP_CACHE")) {
            if (hits < 1000 || ins < 4) {
                fprintf(stderr,
                        "FAIL long: plong hits=%zu inserts=%zu (want hits >= "
                        "1000 and inserts >= 4)\n",
                        hits, ins);
                return 1;
            }
        }
        printf("long %llu %zu\n", h, n);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "--child") == 0) {
        setenv("TSIMD_DISABLE_PCACHE", "1", 1);
        int which = atoi(argv[2]);
        if (which < 0 || which >= (int)(sizeof(k_models) / sizeof(*k_models)))
            return 1;
        unsigned long long h = 0;
        size_t n = 0;
        if (run_one(k_models[which], &h, &n) != 0) return 2;
        printf("%llu %zu\n", h, n);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "--warm") == 0) {
        int which = atoi(argv[2]);
        unsigned long long h = 0;
        size_t n = 0;
        if (run_one(k_models[which], &h, &n) != 0) return 2;
        printf("%llu %zu\n", h, n);
        return 0;
    }

    int fail = 0, checks = 0;
    for (int i = 0; i < (int)(sizeof(k_models) / sizeof(*k_models)); i++) {
        int pipefd[2];
        if (pipe(pipefd) != 0) return 1;
        pid_t pid = fork();
        if (pid == 0) {
            close(pipefd[0]);
            dup2(pipefd[1], STDOUT_FILENO);
            close(pipefd[1]);
            char idx[8];
            snprintf(idx, sizeof idx, "%d", i);
            execl(argv[0], argv[0], "--warm", idx, (char *)NULL);
            _exit(3);
        }
        close(pipefd[1]);
        char out[128];
        ssize_t got = read(pipefd[0], out, sizeof out - 1);
        close(pipefd[0]);
        int status = 0;
        waitpid(pid, &status, 0);
        if (got <= 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "FAIL %s: warm child failed\n", k_models[i]);
            fail = 1;
            continue;
        }
        out[got] = 0;
        unsigned long long hw = 0;
        size_t nw = 0;
        sscanf(out, "%llu %zu", &hw, &nw);

        if (pipe(pipefd) != 0) return 1;
        pid = fork();
        if (pid == 0) {
            close(pipefd[0]);
            dup2(pipefd[1], STDOUT_FILENO);
            close(pipefd[1]);
            char idx[8];
            snprintf(idx, sizeof idx, "%d", i);
            execl(argv[0], argv[0], "--child", idx, (char *)NULL);
            _exit(3);
        }
        close(pipefd[1]);
        got = read(pipefd[0], out, sizeof out - 1);
        close(pipefd[0]);
        status = 0;
        waitpid(pid, &status, 0);
        if (got <= 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "FAIL %s: no-cache child failed\n", k_models[i]);
            fail = 1;
            continue;
        }
        out[got] = 0;
        unsigned long long hn = 0;
        size_t nn = 0;
        sscanf(out, "%llu %zu", &hn, &nn);
        checks++;
        if (hw != hn || nw != nn) {
            fprintf(stderr, "FAIL %s: cache %llu/%zu vs no-cache %llu/%zu\n",
                    k_models[i], hw, nw, hn, nn);
            fail = 1;
        } else {
            printf("%-58s %zu ids, hash equal both ways\n", k_models[i], nw);
        }
    }
    {
        unsigned long long h = 0;
        size_t n = 0, hits = 0, ins = 0;
        if (run_long(&h, &n, &hits, &ins) != 0) {
            fprintf(stderr, "FAIL long corpus: encode failed\n");
            fail = 1;
        } else if (hits < 1000 || ins < 4) {
            fprintf(stderr,
                    "FAIL long corpus: plong hits=%zu inserts=%zu (want hits "
                    ">= 1000 and inserts >= 4)\n",
                    hits, ins);
            fail = 1;
        } else {
            checks++;
            printf("%-58s long corpus %zu ids, plong hits=%zu inserts=%zu\n",
                   "gemma4", n, hits, ins);
        }
    }
    printf("test_spm_cache: %d checks, %s\n", checks, fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}
