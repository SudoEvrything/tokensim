#include "pretokenizer.h"
#include "bpe.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void dump(const char *tag, uint32_t *s, size_t n) {
    printf("%s(%zu):", tag, n);
    for (size_t i = 0; i < n && i < 40; i++) printf(" %u", s[i]);
    printf("%s\n", n > 40 ? " ..." : "");
}
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    static uint8_t d[8192];
    size_t n = fread(d, 1, sizeof d, f);
    fclose(f);
    tsimd_span_array_t a, b;
    tsimd_span_array_init(&a);
    tsimd_span_array_init(&b);
    tsimd_pretokenize_scalar_ref(TSIMD_PRETOK_DEEPSEEK, d, n, &a);
    tsimd_pretokenize_deepseek_masks(d, n, &b);
    printf("scalar %zu spans, masks %zu spans\n", a.count, b.count);
    for (size_t i = 0; i < a.count && i < b.count; i++)
        if (a.spans[i].start != b.spans[i].start || a.spans[i].end != b.spans[i].end)
            printf("DIFF at %zu: scalar [%u,%u) masks [%u,%u)\n", i,
                   a.spans[i].start, a.spans[i].end, b.spans[i].start, b.spans[i].end);
    static tsimd_pcache_span_t ents[16384];
    tsimd_o200k_mstate_t st;
    memset(&st, 0, sizeof st);
    int kn = tsimd_deepseek_mask_fill_keyed(d, n, &st, ents, 16384, NULL, 0);
    printf("keyed fill %d spans\n", kn);
    size_t mism = 0;
    for (int i = 0; i < kn; i++) {
        size_t off = (size_t)(ents[i].ptr - d);
        size_t sl = tsimd_pcache_span_len(&ents[i]);
        if (mism < 4 && (i >= (int)a.count || off != a.spans[i].start ||
                         sl != a.spans[i].end - a.spans[i].start)) {
            printf("KEYED DIFF at %d: keyed [%zu,%zu) scalar [%u,%u)\n", i, off,
                   off + sl, a.spans[i].start, a.spans[i].end);
            mism++;
        }
    }
    tsimd_tokenizer_t tok;
    memset(&tok, 0, sizeof tok);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&tok, TSIMD_MODEL_TOKENIZER_JSON,
                                   "models/reference/deepseek_tokenizer.json",
                                   NULL, &sch, NULL, 0) != 0)
        return 1;
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    tsimd_tokenizer_encode(&tok, d, n, &seq);
    dump("encode", seq.ids, seq.count);
    uint32_t ref[8192];
    size_t rn = 0;
    for (size_t i = 0; i < a.count; i++) {
        uint32_t ids[4096];
        uint32_t len = a.spans[i].end - a.spans[i].start;
        if (len > 200) len = 200;
        for (uint32_t k = 0; k < len; k++)
            ids[k] = tok.vocab.byte_ids[d[a.spans[i].start + k]];
        uint32_t m = len;
        while (m > 1) {
            uint32_t best = UINT32_MAX, bp = 0, bid = 0;
            for (uint32_t k = 0; k + 1 < m; k++) {
                uint32_t id = 0;
                uint32_t p = tsimd_vocab_merge_prio_id(
                    &tok.vocab, ids[k], ids[k + 1], &id);
                if (p != UINT32_MAX && p < best) { best = p; bp = k; bid = id; }
            }
            if (best == UINT32_MAX) break;
            ids[bp] = bid;
            for (uint32_t k = bp + 1; k + 1 < m; k++) ids[k] = ids[k + 1];
            m--;
        }
        for (uint32_t k = 0; k < m; k++) ref[rn++] = ids[k];
    }
    dump("law-bpe", ref, rn);
    int nd = 0;
    for (size_t i = 0; i < seq.count && i < rn; i++)
        if (seq.ids[i] != ref[i]) { printf("ENCODE-DIFF at %zu: %u vs %u\n", i, seq.ids[i], ref[i]); if (++nd > 6) break; }
    printf("encode n=%zu law n=%zu\n", seq.count, rn);
    return 0;
}
