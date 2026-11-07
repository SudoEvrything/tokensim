#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "bpe.h"
#include "pretokenizer.h"

static volatile sig_atomic_t spun = 0;
static size_t g_cand = 0;

static void on_alrm(int sig) {
    (void)sig;
    spun = 1;
    _exit(42);
}

static size_t fill_from(const uint8_t *buf, size_t len, size_t pos,
                        size_t cap, int tp, size_t *total_out) {
    static tsimd_pcache_span_t ents[512];
    tsimd_o200k_mstate_t st;
    memset(&st, 0, sizeof st);
    st.pos = pos;
    st.scan = 64;
    size_t total = 0;
    for (;;) {
        alarm(3);
        int n = tsimd_nemo_mask_fill_keyed(buf, len, &st, ents, cap, NULL,
                                           tp);
        alarm(0);
        if (n < 0) return (size_t)-1;
        if (n == 0) break;
        total += (size_t)n;
    }
    *total_out = total;
    return total;
}

int main(void) {
    signal(SIGALRM, on_alrm);
    static const struct {
        const char *filler;
    } fills[] = {{"a."}, {"z,"}, {"q x"}, {"b;c"}};
    static const struct {
        const uint8_t b[4];
        size_t n;
    } bads[] = {
        {{0xe4, 0xb8, 0xad}, 3}, {{0xc3, 0xa9}, 2}, {{0x8c}, 1},
        {{0xf0, 0x9f, 0x98, 0x80}, 4},
    };
    static const size_t caps[] = {64, 128, 192};
    uint8_t buf[320];
    size_t cand = 0;
    for (size_t f = 0; f < sizeof(fills) / sizeof(fills[0]); f++) {
        for (size_t b = 0; b < sizeof(bads) / sizeof(bads[0]); b++) {
            for (size_t off = 124; off <= 126; off++) {
                for (size_t c = 0; c < sizeof(caps) / sizeof(caps[0]); c++) {
                    size_t len = 0;
                    while (len < off) {
                        const char *fl = fills[f].filler;
                        size_t fln = strlen(fl);
                        buf[len++] = (uint8_t)fl[len % fln];
                    }
                    memcpy(buf + len, bads[b].b, bads[b].n);
                    len += bads[b].n;
                    while (len < 300) {
                        const char *fl = fills[f].filler;
                        size_t fln = strlen(fl);
                        buf[len++] = (uint8_t)fl[len % fln];
                    }
                    tsimd_span_array_t a;
                    tsimd_span_array_init(&a);
                    if (tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_NEMO, buf,
                                                     len, &a) != 0)
                        continue;
                    for (size_t i = 0; i < a.count; i++) {
                        size_t s = a.spans[i].start;
                        size_t e = a.spans[i].end;
                        if (s == 0 || s >= 64 || e < 64) continue;
                        for (int tp = 0; tp <= 1; tp++) {
                            cand++;
                            g_cand = cand;
                            size_t total = 0;
                            size_t r = fill_from(buf, len, s, caps[c], tp,
                                                 &total);
                            if (spun) {
                                printf("SPIN cand=%zu f=%zu b=%zu off=%zu "
                                       "cap=%zu tp=%d pos=%zu\n",
                                       cand, f, b, off, caps[c], tp, s);
                                fflush(stdout);
                                return 42;
                            }
                            if (r == (size_t)-1) continue;
                        }
                    }
                    tsimd_span_array_free(&a);
                }
            }
        }
    }
    printf("NO SPIN in %zu candidates\n", cand);
    return 0;
}
