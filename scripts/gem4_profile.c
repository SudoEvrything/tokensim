#include "tokenizer.h"
#include "pretokenizer.h"
#include "unicode.h"

#include "../src/model_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <emmintrin.h>

static tsimd_tokenizer_t tok;
static tsimd_bpe_workspace_t ws;

static size_t (*ms_fn)(const uint8_t *, size_t, uint8_t *, size_t, int);

size_t ms_impl_old(const uint8_t *, size_t, uint8_t *, size_t, int);

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t fnv(const uint32_t *ids, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= ids[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static void emit_id(uint32_t id) {
    if (tsimd_ws_out_ensure(&ws, ws.out_count + 1) != 0) exit(2);
    if (!ws.out) {
        fprintf(stderr, "emit: ws.out NULL at count=%zu\n", ws.out_count);
        exit(3);
    }
    ws.out[ws.out_count++] = id;
}

typedef struct {
    uint64_t seams, meta, bpe, emit;
    size_t nseg, seg_bytes, seg_min, seg_max;
} buckets_t;

static uint64_t seam_split[2];

static void seam_cost_split(const uint8_t *data, size_t len) {
    uint64_t t0 = now_ns();
    tsimd_added_seam_list_t seams;
    if (tsimd_tokenizer_added_seams(&tok, data, len, &seams) != 0) return;
    uint64_t full = now_ns() - t0;
    tsimd_added_seam_list_free(&seams);
    uint8_t first[256] = {0};
    for (size_t s = 0; s < tok.added_count; s++)
        if (tok.added[s].length > 0)
            first[(uint8_t)tok.added[s].text[0]] = 1;
    uint8_t fb[8];
    int nfb = 0;
    for (int b = 0; b < 256 && nfb < 8; b++)
        if (first[b]) fb[nfb++] = (uint8_t)b;
    __m128i vb[8];
    for (int k = 0; k < nfb; k++) vb[k] = _mm_set1_epi8((char)fb[k]);
    size_t pos = 0, cands = 0, calls = 0;
    uint64_t t2 = now_ns();
    while (pos + 16 <= len) {
        __m128i v =
            _mm_loadu_si128((const __m128i *)(const void *)(data + pos));
        __m128i m = _mm_cmpeq_epi8(v, vb[0]);
        for (int k = 1; k < nfb; k++) m = _mm_or_si128(m, _mm_cmpeq_epi8(v, vb[k]));
        unsigned mask = (unsigned)_mm_movemask_epi8(m);
        cands += (unsigned)__builtin_popcount(mask);
        while (mask) {
            unsigned k2 = (unsigned)__builtin_ctz(mask);
            size_t cand = pos + k2;
            size_t idx = 0;
            calls++;
            if (tsimd_tokenizer_added_len_at(&tok, data, len, cand, &idx)) {
                pos = cand + idx;
                goto next_blk;
            }
            mask &= mask - 1;
        }
        pos += 16;
    next_blk:;
    }
    uint64_t loop = now_ns() - t2;
    seam_split[0] = full;
    seam_split[1] = full - loop;
    fprintf(stderr,
            "seam split: full %.2f ms | cmp-loop %.2f ms | len_at %.2f ms "
            "(cands=%zu len_at_calls=%zu seams=%zu nfb=%d)\n",
            (double)full / 1e6, (double)loop / 1e6,
            (double)(full - loop) / 1e6, cands, calls, seams.count, nfb);
}

static int mirror_batched(const uint8_t *data, size_t len, buckets_t *bk,
                          uint64_t *hash, size_t *count) {
    uint64_t t0 = now_ns();
    tsimd_added_seam_list_t seams;
    if (tsimd_tokenizer_added_seams(&tok, data, len, &seams) != 0) return -1;
    bk->seams += now_ns() - t0;
    size_t nseg = seams.count + 1;
    size_t *hard = malloc(nseg * sizeof(size_t));
    uint32_t *added_ids = malloc((seams.count ? seams.count : 1) *
                                 sizeof(uint32_t));
    size_t cap = len * 3 + 16;
    uint8_t *norm = malloc(cap);
    if (!hard || !added_ids || !norm) return -1;
    size_t noff = 0, seg = 0, nseg_real = 0;
    for (size_t i = 0; i < seams.count; i++) {
        const tsimd_added_seam_t *sm = &seams.items[i];
        if (sm->off > seg) {
            uint64_t t1 = now_ns();
            size_t nn = ms_fn(data + seg, sm->off - seg,
                  norm + noff, cap - noff, 0);
            bk->meta += now_ns() - t1;
            if (nn == (size_t)-1) return -1;
            noff += nn;
        }
        hard[nseg_real++] = noff;
        added_ids[i] = tok.added[sm->idx].id;
        seg = sm->off + sm->len;
    }
    if (len > seg) {
        uint64_t t1 = now_ns();
        size_t nn = ms_fn(data + seg, len - seg, norm + noff,
                  cap - noff, 0);
        bk->meta += now_ns() - t1;
        if (nn == (size_t)-1) return -1;
        noff += nn;
    }
    hard[nseg_real++] = noff;
    uint32_t target = tsimd_spm_word_target(&tok.vocab, 0);
    for (size_t i = 0; i < nseg_real; i++) {
        size_t seglen = hard[i] - (i ? hard[i - 1] : 0);
        if (seglen) {
            bk->nseg++;
            bk->seg_bytes += seglen;
            if (seglen < bk->seg_min || bk->seg_min == 0) bk->seg_min = seglen;
            if (seglen > bk->seg_max) bk->seg_max = seglen;
            uint64_t t2 = now_ns();
            if (tsimd_bpe_encode_spm_blocked_cached(
                    &tok.vocab, &tok.cache, &tok.cache, &ws, norm +
                    (i ? hard[i - 1] : 0), (uint32_t)seglen, target) != 0)
                return -1;
            bk->bpe += now_ns() - t2;
        }
        if (i < seams.count) {
            uint64_t t3 = now_ns();
            emit_id(added_ids[i]);
            bk->emit += now_ns() - t3;
        }
    }
    if (hash) *hash = fnv(ws.out, ws.out_count);
    if (count) *count = ws.out_count;
    free(hard);
    free(added_ids);
    free(norm);
    tsimd_added_seam_list_free(&seams);
    return 0;
}

static int mirror_sequential(const uint8_t *data, size_t len, buckets_t *bk,
                             uint64_t *hash, size_t *count) {
    uint64_t t0 = now_ns();
    tsimd_added_seam_list_t seams;
    if (tsimd_tokenizer_added_seams(&tok, data, len, &seams) != 0) return -1;
    bk->seams += now_ns() - t0;
    size_t seg = 0;
    for (size_t i = 0; i < seams.count; i++) {
        const tsimd_added_seam_t *sm = &seams.items[i];
        if (sm->off > seg) {
            size_t slen = sm->off - seg;
            size_t cap = slen * 3 + 16;
            uint8_t *norm = malloc(cap);
            if (!norm) return -1;
            uint64_t t1 = now_ns();
            size_t nn = ms_fn(data + seg, slen, norm, cap, 0);
            bk->meta += now_ns() - t1;
            if (nn == (size_t)-1) return -1;
            bk->nseg++;
            bk->seg_bytes += nn;
            if (nn < bk->seg_min || bk->seg_min == 0) bk->seg_min = nn;
            if (nn > bk->seg_max) bk->seg_max = nn;
            uint64_t t2 = now_ns();
            if (tsimd_bpe_encode_spm_blocked_cached(
                    &tok.vocab, &tok.cache, &tok.cache, &ws, norm,
                    (uint32_t)nn,
                    tsimd_spm_word_target(&tok.vocab, 0)) != 0)
                return -1;
            bk->bpe += now_ns() - t2;
            free(norm);
        }
        uint64_t t3 = now_ns();
        if (tok.added[sm->idx].length > 0) emit_id(tok.added[sm->idx].id);
        bk->emit += now_ns() - t3;
        seg = sm->off + sm->len;
    }
    if (len > seg) {
        size_t slen = len - seg;
        size_t cap = slen * 3 + 16;
        uint8_t *norm = malloc(cap);
        if (!norm) return -1;
        uint64_t t1 = now_ns();
        size_t nn = ms_fn(data + seg, slen, norm, cap, 0);
        bk->meta += now_ns() - t1;
        if (nn == (size_t)-1) return -1;
        bk->nseg++;
        bk->seg_bytes += nn;
        uint64_t t2 = now_ns();
        if (tsimd_bpe_encode_spm_blocked_cached(&tok.vocab, &tok.cache,
                                                &tok.cache, &ws, norm,
                                                (uint32_t)nn,
                                                tsimd_spm_word_target(
                                                    &tok.vocab, 0)) != 0)
            return -1;
        bk->bpe += now_ns() - t2;
        free(norm);
    }
    if (hash) *hash = fnv(ws.out, ws.out_count);
    if (count) *count = ws.out_count;
    tsimd_added_seam_list_free(&seams);
    return 0;
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1]
                                 : "models/reference/gemma3-4b_tokenizer.json";
    const char *corpus = argc > 2 ? argv[2] : "testdata/owt16m.txt";
    int rounds = argc > 3 ? atoi(argv[3]) : 3;
    ms_fn = (argc > 4 && strcmp(argv[4], "old") == 0) ? ms_impl_old
                                                      : tsimd_metaspace_ex;
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GEM4;
    if (tsimd_tokenizer_load_model(&tok, TSIMD_MODEL_TOKENIZER_JSON, model,
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 2;
    }
    if (tsimd_bpe_workspace_init(&ws, tok.vocab.max_token_len) != 0) return 2;
    FILE *rf = fopen(corpus, "rb");
    if (!rf) {
        fprintf(stderr, "open %s\n", corpus);
        return 2;
    }
    fseek(rf, 0, SEEK_END);
    long fsz = ftell(rf);
    fseek(rf, 0, SEEK_SET);
    if (fsz <= 0) return 2;
    size_t len = (size_t)fsz;
    uint8_t *data = malloc(len);
    if (!data || fread(data, 1, len, rf) != len) return 2;
    fclose(rf);
    double mb = (double)len / (1024.0 * 1024.0);
    for (int r = 0; r < rounds; r++) {
        buckets_t bb;
        memset(&bb, 0, sizeof bb);
        fprintf(stderr, "dbg: ws.out=%p cap=%zu\n", (void *)ws.out,
                (size_t)ws.out_capacity);
        ws.out_count = 0;
        uint64_t w0 = now_ns();
        if (mirror_batched((const uint8_t *)data, len, &bb, NULL, NULL) != 0)
            return 2;
        uint64_t wtot = now_ns() - w0;
        buckets_t sb;
        memset(&sb, 0, sizeof sb);
        ws.out_count = 0;
        uint64_t s0 = now_ns();
        if (mirror_sequential((const uint8_t *)data, len, &sb, NULL, NULL) !=
            0)
            return 2;
        uint64_t stot = now_ns() - s0;
        if (r == rounds - 1) {
            seam_cost_split((const uint8_t *)data, len);
            printf("slice %.1f MiB\n", mb);
            printf("batched:   total %.1f MB/s (%.1f ms) | seams %.1f%% "
                   "(%.1f ms) meta %.1f%% (%.1f ms) bpe %.1f%% (%.1f ms) "
                   "emit %.1f%% | segs %zu mean %.0f B (min %zu max %zu)\n",
                   (double)len * 1e9 / (double)wtot / 1048576.0,
                   (double)wtot / 1e6,
                   100.0 * (double)bb.seams / (double)wtot,
                   (double)bb.seams / 1e6,
                   100.0 * (double)bb.meta / (double)wtot,
                   (double)bb.meta / 1e6,
                   100.0 * (double)bb.bpe / (double)wtot,
                   (double)bb.bpe / 1e6,
                   100.0 * (double)bb.emit / (double)wtot, bb.nseg,
                   bb.nseg ? (double)bb.seg_bytes / (double)bb.nseg : 0.0,
                   bb.seg_min, bb.seg_max);
            tsimd_bpe_split_dump();
            printf("seq:       total %.1f MB/s (%.1f ms) | seams %.1f%% "
                   "(%.1f ms) meta %.1f%% (%.1f ms) bpe %.1f%% (%.1f ms) "
                   "emit %.1f%% | segs %zu mean %.0f B\n",
                   (double)len * 1e9 / (double)stot / 1048576.0,
                   (double)stot / 1e6,
                   100.0 * (double)sb.seams / (double)stot,
                   (double)sb.seams / 1e6,
                   100.0 * (double)sb.meta / (double)stot,
                   (double)sb.meta / 1e6,
                   100.0 * (double)sb.bpe / (double)stot,
                   (double)sb.bpe / 1e6,
                   100.0 * (double)sb.emit / (double)stot, sb.nseg,
                   sb.nseg ? (double)sb.seg_bytes / (double)sb.nseg : 0.0);
        }
    }
    return 0;
}
