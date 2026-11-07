#include "bpe.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

static int fails = 0;
static int checks = 0;

#define CK(cond, ...)                                  \
    do {                                               \
        checks++;                                      \
        if (!(cond)) {                                 \
            fprintf(stderr, "FAIL %d: ", __LINE__);    \
            fprintf(stderr, __VA_ARGS__);              \
            fprintf(stderr, "\n");                     \
            fails++;                                   \
        }                                              \
    } while (0)

typedef struct {
    int rc;
    size_t count;
    unsigned long long hash;
    long peak_kb;
} child_result_t;

static size_t fill_input(uint8_t *buf, size_t n) {
    FILE *f = fopen(TSIMD_SRC_DIR "/testdata/owt.txt", "rb");
    if (!f) return 0;
    size_t got = fread(buf, 1, n, f);
    fclose(f);
    return got;
}

static size_t fill_cjk(uint8_t *buf, size_t n) {
    const char *unit = "数字123と漢字のテスト、日本語のテキスト。 héllo wörld ";
    size_t ul = strlen(unit);
    size_t off = 0;
    while (off < n) {
        size_t take = n - off < ul ? n - off : ul;
        memcpy(buf + off, unit, take);
        off += take;
    }
    return n;
}

static size_t fill_owt16(uint8_t *buf, size_t n) {
    FILE *f = fopen(TSIMD_SRC_DIR "/testdata/owt16m.txt", "rb");
    if (!f) return 0;
    size_t got = fread(buf, 1, n, f);
    fclose(f);
    return got;
}

static size_t fill_cutclass(uint8_t *buf) {
    const size_t off = 1940000u;
    const size_t fn = 24000u;
    const size_t tail = 2000000u;
    fill_cjk(buf, off);
    memset(buf + off, 0xFF, fn);
    fill_cjk(buf + off + fn, tail);
    return off + fn + tail;
}

static void child_run(int mode, const uint8_t *buf, size_t n,
                      int wfd) {
    if (mode == 1)
        setenv("TSIMD_DISABLE_UNIGRAM_MEMO", "1", 1);
    else
        setenv("TSIMD_ENABLE_UNIGRAM_MEMO", "1", 1);
    if (mode == 5 || mode == 7 || mode == 10)
        setenv("TSIMD_UNI_WINDOW", "1", 1);
    if (mode == 8)
        setenv("TSIMD_UNI_WINDOW", "0", 1);
    child_result_t r;
    memset(&r, 0, sizeof r);
    tsimd_tokenizer_t tok;
    memset(&tok, 0, sizeof tok);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(
            &tok, TSIMD_MODEL_SENTENCEPIECE,
            TSIMD_SRC_DIR "/models/reference/t5_spiece.model",
            NULL, &sch, NULL, 0) != 0) {
        r.rc = -1;
        if (write(wfd, &r, sizeof r) != (ssize_t)sizeof r) _exit(2);
        _exit(1);
    }
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    if (mode == 2 || mode == 4)
        r.rc = tsimd_tokenizer_encode_parallel(&tok, buf, n, 6, &seq);
    else if (mode == 5 || mode == 7 || mode == 8 || mode == 10)
        r.rc = tsimd_tokenizer_encode_whole_chain_par(&tok, buf, n, 6, &seq);
    else
        r.rc = tsimd_tokenizer_encode(&tok, buf, n, &seq);
    if (r.rc == 0) {
        r.count = seq.count;
        unsigned long long h = 1469598103934665603ULL;
        for (size_t i = 0; i < seq.count; i++) {
            h ^= seq.ids[i];
            h *= 1099511628211ULL;
        }
        r.hash = h;
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    r.peak_kb = (long)ru.ru_maxrss;
    tsimd_token_sequence_free(&seq);
    tsimd_tokenizer_free(&tok);
    if (write(wfd, &r, sizeof r) != (ssize_t)sizeof r) _exit(2);
    _exit(r.rc == 0 ? 0 : 1);
}

static int run_mode(int mode, const uint8_t *buf, size_t n,
                    child_result_t *out) {
    int pfd[2];
    if (pipe(pfd) != 0) return -2;
    pid_t p = fork();
    if (p < 0) return -2;
    if (p == 0) {
        close(pfd[0]);
        child_run(mode, buf, n, pfd[1]);
        _exit(3);
    }
    close(pfd[1]);
    child_result_t r;
    memset(&r, 0xFF, sizeof r);
    size_t off = 0;
    char *dst = (char *)&r;
    while (off < sizeof r) {
        ssize_t got = read(pfd[0], dst + off, sizeof r - off);
        if (got <= 0) break;
        off += (size_t)got;
    }
    close(pfd[0]);
    int st = 0;
    waitpid(p, &st, 0);
    *out = r;
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

int main(void) {
    size_t n = 128u << 20;
    uint8_t *buf = malloc(n);
    if (!buf) return 1;
    n = fill_input(buf, n);
    if (n < (64u << 20)) {
        fprintf(stderr, "SKIP: testdata/owt.txt missing or small\n");
        free(buf);
        return 0;
    }
    child_result_t memo_r, fb_r, par_r;

    size_t cn = 4698000u;
    uint8_t *cbuf = malloc(cn);
    if (!cbuf) return 1;
    fill_cjk(cbuf, cn);
    child_result_t c1_r, c6_r, c5_r, c8_r;

    int rc = run_mode(3, cbuf, cn, &c1_r);
    CK(rc == 0 && c1_r.rc == 0, "cjk 1t encode rc=%d/%d", rc, c1_r.rc);
    rc = run_mode(4, cbuf, cn, &c6_r);
    CK(rc == 0 && c6_r.rc == 0, "cjk 6t encode rc=%d/%d", rc, c6_r.rc);
    if (c1_r.rc == 0 && c6_r.rc == 0) {
        CK(c1_r.count == c6_r.count, "cjk count mismatch %zu vs %zu",
           c1_r.count, c6_r.count);
        CK(c1_r.hash == c6_r.hash, "cjk fingerprint mismatch");
    }
    printf("cjk: 1t=%zu 6t=%zu %s\n", c1_r.count, c6_r.count,
           c1_r.hash == c6_r.hash ? "match" : "MISMATCH");

    rc = run_mode(5, cbuf, cn, &c5_r);
    CK(rc == 0 && c5_r.rc == 0, "cjk windowed-6t encode rc=%d/%d", rc,
       c5_r.rc);
    if (c1_r.rc == 0 && c5_r.rc == 0) {
        CK(c1_r.count == c5_r.count, "cjk windowed count mismatch %zu vs %zu",
           c1_r.count, c5_r.count);
        CK(c1_r.hash == c5_r.hash, "cjk windowed fingerprint mismatch");
    }
    printf("cjk windowed: 1t=%zu w6t=%zu %s\n", c1_r.count, c5_r.count,
           c1_r.hash == c5_r.hash ? "match" : "MISMATCH");

    rc = run_mode(8, cbuf, cn, &c8_r);
    CK(rc == 0 && c8_r.rc == 0, "cjk whole-escape encode rc=%d/%d", rc,
       c8_r.rc);
    if (c1_r.rc == 0 && c8_r.rc == 0) {
        CK(c1_r.count == c8_r.count, "cjk whole-escape count mismatch");
        CK(c1_r.hash == c8_r.hash, "cjk whole-escape fingerprint mismatch");
    }
    printf("cjk whole-escape: 1t=%zu e6t=%zu %s\n", c1_r.count, c8_r.count,
           c1_r.hash == c8_r.hash ? "match" : "MISMATCH");
    free(cbuf);

    size_t on = 16u << 20;
    uint8_t *obuf = malloc(on);
    if (!obuf) return 1;
    on = fill_owt16(obuf, on);
    if (on >= (8u << 20)) {
        child_result_t o1_r, o7_r;
        rc = run_mode(6, obuf, on, &o1_r);
        CK(rc == 0 && o1_r.rc == 0, "owt16 seq encode rc=%d/%d", rc, o1_r.rc);
        rc = run_mode(7, obuf, on, &o7_r);
        CK(rc == 0 && o7_r.rc == 0, "owt16 windowed encode rc=%d/%d", rc,
           o7_r.rc);
        if (o1_r.rc == 0 && o7_r.rc == 0) {
            CK(o1_r.count == o7_r.count,
               "owt16 windowed count mismatch %zu vs %zu", o1_r.count,
               o7_r.count);
            CK(o1_r.hash == o7_r.hash, "owt16 windowed fingerprint mismatch");
        }
        printf("owt16: seq=%zu win=%zu %s\n", o1_r.count, o7_r.count,
               o1_r.hash == o7_r.hash ? "match" : "MISMATCH");
    } else {
        printf("SKIP: testdata/owt16m.txt missing or small\n");
    }
    free(obuf);

    size_t xn = 3964000u;
    uint8_t *xbuf = malloc(xn);
    if (!xbuf) return 1;
    xn = fill_cutclass(xbuf);
    child_result_t x1_r, xa_r;
    rc = run_mode(9, xbuf, xn, &x1_r);
    CK(rc == 0 && x1_r.rc == 0, "cutclass seq encode rc=%d/%d", rc, x1_r.rc);
    rc = run_mode(10, xbuf, xn, &xa_r);
    CK(rc == 0 && xa_r.rc == 0, "cutclass windowed encode rc=%d/%d", rc,
       xa_r.rc);
    if (x1_r.rc == 0 && xa_r.rc == 0) {
        CK(x1_r.count == xa_r.count,
           "cutclass windowed count mismatch %zu vs %zu", x1_r.count,
           xa_r.count);
        CK(x1_r.hash == xa_r.hash, "cutclass windowed fingerprint mismatch");
    }
    printf("cutclass: seq=%zu win=%zu %s\n", x1_r.count, xa_r.count,
           x1_r.hash == xa_r.hash ? "match" : "MISMATCH");
    free(xbuf);

    rc = run_mode(2, buf, n, &par_r);
    CK(rc == 0 && par_r.rc == 0, "parallel-6t encode rc=%d/%d", rc, par_r.rc);
    printf("parallel-6t: count=%zu peak=%ld MB\n", par_r.count,
           par_r.peak_kb / 1024);
    if (rc == 0 && par_r.rc == 0)
        CK(par_r.peak_kb > 0 && par_r.peak_kb < 3200000,
           "parallel-6t peak %ld KB", par_r.peak_kb);

    rc = run_mode(0, buf, n, &memo_r);
    CK(rc == 0 && memo_r.rc == 0, "memo-on encode rc=%d/%d", rc, memo_r.rc);
    printf("memo-on: count=%zu peak=%ld MB\n", memo_r.count,
           memo_r.peak_kb / 1024);
    if (rc == 0 && memo_r.rc == 0)
        CK(memo_r.peak_kb > 0 && memo_r.peak_kb < 1600000,
           "memo-on peak %ld KB", memo_r.peak_kb);

    rc = run_mode(1, buf, n, &fb_r);
    CK(rc == 0 && fb_r.rc == 0, "fallback encode rc=%d/%d", rc, fb_r.rc);
    printf("fallback: count=%zu peak=%ld MB\n", fb_r.count,
           fb_r.peak_kb / 1024);
    if (rc == 0 && fb_r.rc == 0)
        printf("fallback peak %ld KB (the whole-buffer path, unbounded)\n",
               fb_r.peak_kb);

    if (memo_r.rc == 0 && fb_r.rc == 0) {
        CK(memo_r.count == fb_r.count, "count mismatch %zu vs %zu",
           memo_r.count, fb_r.count);
        CK(memo_r.hash == fb_r.hash, "id fingerprint mismatch");
    }
    if (memo_r.rc == 0 && par_r.rc == 0) {
        CK(memo_r.count == par_r.count, "par count mismatch %zu vs %zu",
           memo_r.count, par_r.count);
        CK(memo_r.hash == par_r.hash, "par id fingerprint mismatch");
    }

    free(buf);
    printf("test_unigram_chunk: %d checks, %d fails\n", checks, fails);
    return fails != 0;
}
