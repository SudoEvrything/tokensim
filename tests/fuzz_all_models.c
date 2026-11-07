#include "tokenizer.h"
#include "pretokenizer.h"
#include "unicode.h"

#include "../src/model_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "spm_tie_rule.h"

static tsimd_tokenizer_t tok;
static int ready;
static int spm_mode;
static int uni_mode;
static tsimd_pretok_scheme_t sch;

static uint32_t ref_ids[4096];
static size_t ref_n;

static void naive_bpe_span(const uint8_t *bytes, uint32_t len) {
    uint32_t ids[4096];
    if (len > 200) len = 200;
    if (getenv("TSIMD_FUZZ_DBG")) {
        fprintf(stderr, "bytes:");
        for (uint32_t i = 0; i < len; i++)
            fprintf(stderr, " %u/%u", bytes[i], tok.vocab.byte_ids[bytes[i]]);
        fprintf(stderr, "\nprios:");
        for (uint32_t i = 0; i + 1 < len; i++) {
            uint32_t id = 0;
            fprintf(stderr, " %u", tsimd_vocab_merge_prio_id(
                                       &tok.vocab, tok.vocab.byte_ids[bytes[i]],
                                       tok.vocab.byte_ids[bytes[i + 1]], &id));
        }
        fprintf(stderr, "\n");
    }
    for (uint32_t i = 0; i < len; i++)
        ids[i] = tok.vocab.byte_ids[bytes[i]];
    uint32_t n = len;
    while (n > 1) {
        uint32_t best = UINT32_MAX, best_pos = 0, best_id = 0;
        for (uint32_t k = 0; k + 1 < n; k++) {
            uint32_t id = 0;
            uint32_t p = tsimd_vocab_merge_prio_id(&tok.vocab, ids[k],
                                                   ids[k + 1], &id);
            if (p != UINT32_MAX && p < best) {
                best = p;
                best_pos = k;
                best_id = id;
            }
        }
        if (best == UINT32_MAX) break;
        ids[best_pos] = best_id;
        for (uint32_t k = best_pos + 1; k + 1 < n; k++)
            ids[k] = ids[k + 1];
        n--;
    }
    for (uint32_t i = 0; i < n && ref_n < 4096; i++)
        ref_ids[ref_n++] = ids[i];
}

static void naive_gem4_span(const uint8_t *bytes, uint32_t len) {
    uint32_t ids[512];
    uint32_t n = 0;
    for (uint32_t i = 0; i < len && n < 500;) {
        uint32_t cp;
        tsimd_char_class_t cls;
        size_t a = tsimd_utf8_next(bytes + i, len - i, &cp, &cls);
        if (a == 0) a = 1;
        uint32_t id = a == 1 ? tok.vocab.byte_ids[bytes[i]]
                             : tsimd_vocab_find(&tok.vocab, bytes + i,
                                                (uint32_t)a);
        if (id == UINT32_MAX) {
            for (size_t k = 0; k < a && n < 511; k++)
                ids[n++] = tok.vocab.byte_ids[bytes[i + k]];
        } else {
            ids[n++] = id;
        }
        i += (uint32_t)a;
    }
    while (n > 1) {
        uint32_t best = UINT32_MAX, bp = 0, bid = 0;
        for (uint32_t k = 0; k + 1 < n; k++) {
            uint32_t mid = 0;
            uint32_t p = tsimd_vocab_merge_prio_id(&tok.vocab, ids[k],
                                                   ids[k + 1], &mid);
            if (p != UINT32_MAX && p < best) {
                best = p;
                bp = k;
                bid = mid;
            }
        }
        if (best == UINT32_MAX) break;
        ids[bp] = bid;
        for (uint32_t k = bp + 1; k + 1 < n; k++) ids[k] = ids[k + 1];
        n--;
    }
    for (uint32_t k = 0; k < n && ref_n < 4096; k++)
        ref_ids[ref_n++] = ids[k];
}

static void naive_spm_span(const uint8_t *bytes, uint32_t len) {
    if (getenv("TSIMD_FUZZ_DBG")) {
        fprintf(stderr, "VIT len=%u bytes:", len);
        for (uint32_t i = 0; i < len && i < 8; i++)
            fprintf(stderr, " %02x", bytes[i]);
        fprintf(stderr, "\n");
    }
    double dp[1025];
    int pre[1025];
    uint32_t ref_tmp[1025];
    memset(ref_tmp, 0xFF, sizeof ref_tmp);
    double unk_sc = -10.0;
    for (uint32_t id = 0; id < tok.vocab.vocab_size; id++) {
        if (tok.vocab.byte_tok &&
            (tok.vocab.byte_tok[id >> 3] >> (id & 7)) & 1)
            continue;
        if (tok.vocab.scores && (double)tok.vocab.scores[id] < unk_sc)
            unk_sc = (double)tok.vocab.scores[id];
    }
    unk_sc -= 10.0;
    static uint16_t unk_width[1025];
    uint8_t covered[1025];
    {
        memset(covered, 0, sizeof covered);
        for (uint32_t p = 0; p < len; p++) {
            for (uint32_t l = 1; l <= len - p && l <= 32; l++)
                if (tsimd_vocab_find(&tok.vocab, bytes + p, l) !=
                    UINT32_MAX) {
                    memset(covered + p, 1, l);
                    break;
                }
        }
        for (uint32_t p = 0; p < len;) {
            if (covered[p]) {
                unk_width[p] = 0;
                p++;
                continue;
            }
            uint32_t q = p;
            while (q < len && !covered[q]) q++;
            for (uint32_t k = p; k < q; k++) unk_width[k] = 0;
            unk_width[p] = (uint16_t)(q - p);
            p = q;
        }
        if (getenv("TSIMD_FUZZ_DBG")) {
            fprintf(stderr, "ORACLE len=%u covered:", len);
            for (uint32_t p = 0; p < len; p++)
                fprintf(stderr, " %u", covered[p]);
            fprintf(stderr, " width:");
            for (uint32_t p = 0; p < len; p++)
                fprintf(stderr, " %u", unk_width[p]);
            fprintf(stderr, "\n");
        }
    }
    dp[0] = 0.0f;
    pre[0] = -1;
    if (getenv("TSIMD_FUZZ_DBG")) {
        for (uint32_t j = 0; j < len && j < 8; j++) {
            for (uint32_t l = 1; l <= len - j && l <= 6; l++) {
                uint32_t fid = tsimd_vocab_find(&tok.vocab, bytes + j, l);
                if (fid != UINT32_MAX)
                    fprintf(stderr, "EDGE j=%u l=%u id=%u sc=%f\n", j, l,
                            fid, tok.vocab.scores ? (double)tok.vocab.scores[fid]
                                                  : -999.0);
            }
        }
        fprintf(stderr, "UNKSC %f\n", (double)unk_sc);
    }
    for (uint32_t i = 1; i <= len; i++) {
        dp[i] = -1e30;
        pre[i] = -1;
        for (uint32_t j = 0; j < i; j++) {
            if (dp[j] <= -1e30) continue;
            uint32_t id = tsimd_vocab_find(&tok.vocab, bytes + j, i - j);
            double sc;
            if (id != UINT32_MAX && id < tok.vocab.vocab_size &&
                tok.vocab.scores &&
                (!tok.vocab.byte_tok ||
                 !((tok.vocab.byte_tok[id >> 3] >> (id & 7)) & 1))) {
                sc = (double)tok.vocab.scores[id];
            } else if (i - j == unk_width[j]) {
                id = tok.vocab.spm_unk_id;
                sc = unk_sc;
            } else {
                continue;
            }
            double cand = dp[j] + sc;
            if (cand > dp[i] + 1e-9 ||
                (cand >= dp[i] - 1e-9 && (int)j > pre[i])) {
                dp[i] = cand;
                pre[i] = (int)j;
                ref_tmp[i] = id;
            }
        }
    }
    uint32_t stack[1025];
    size_t ns = 0;
    int32_t cur = (int32_t)len;
    if (getenv("TSIMD_FUZZ_DBG") && dp[len] <= -1e30) {
        fprintf(stderr, "DEADEND len=%u dp:", len);
        for (uint32_t p = 0; p <= len && p < 40; p++)
            fprintf(stderr, " %d", dp[p] <= -1e30 ? -1 : 0);
        fprintf(stderr, " cov3:");
        for (uint32_t p = 3; p < 9; p++)
            fprintf(stderr, " %u/%u", covered[p], unk_width[p]);
        fprintf(stderr, "\n");
    }
    while (cur > 0 && ns < 1025) {
        stack[ns++] = ref_tmp[cur];
        cur = pre[cur];
    }
    for (size_t k = ns; k > 0 && ref_n < 4096; k--)
        ref_ids[ref_n++] = stack[k - 1];
}

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
    const char *spec = getenv("TSIMD_FUZZ_SPEC");
    if (!spec) return 0;
    char path[512];
    strncpy(path, spec, sizeof path - 1);
    path[sizeof path - 1] = 0;
    char *semi = strchr(path, ';');
    if (!semi) return 0;
    *semi = 0;
    const char *kind = semi + 1;
    char *semi2 = strchr(semi + 1, ';');
    const char *scheme = NULL;
    if (semi2) {
        *semi2 = 0;
        scheme = semi2 + 1;
    }
    memset(&tok, 0, sizeof tok);
    tsimd_model_format_t fmt = strcmp(kind, "spm") == 0
                                   ? TSIMD_MODEL_SENTENCEPIECE
                                   : strcmp(kind, "tiktoken") == 0
                                         ? TSIMD_MODEL_TIKTOKEN
                                         : TSIMD_MODEL_TOKENIZER_JSON;
    sch = TSIMD_PRETOK_GPT2;
    if (scheme) {
        if (strcmp(scheme, "o200k") == 0) sch = TSIMD_PRETOK_O200K;
        else if (strcmp(scheme, "cl100k") == 0) sch = TSIMD_PRETOK_CL100K;
        else if (strcmp(scheme, "kimi") == 0) sch = TSIMD_PRETOK_KIMI;
        else if (strcmp(scheme, "gem4") == 0) sch = TSIMD_PRETOK_GEM4;
    }
    if (tsimd_tokenizer_load_model(&tok, fmt, path, NULL, &sch, NULL,
                                   0) != 0) {
        fprintf(stderr, "FUZZ: model load failed: %s (%s)\n", path, kind);
        return 0;
    }
    spm_mode = fmt == TSIMD_MODEL_SENTENCEPIECE;
    uni_mode = sch == TSIMD_PRETOK_GEM4;
    ready = 1;
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (!ready || size < 2 || size > 200) return 0;
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    if (tsimd_tokenizer_encode(&tok, data, size, &seq) != 0) return 0;
    if (seq.count > 500) {
        tsimd_token_sequence_free(&seq);
        return 0;
    }
    uint32_t mine[512];
    memcpy(mine, seq.ids, seq.count * 4);
    size_t mine_n = seq.count;
    tsimd_token_sequence_free(&seq);

    size_t nlen = 0;
    uint8_t *nd = tsimd_nfc_normalize_all(&tok, data, size, 1, &nlen);
    const uint8_t *buf = nd ? nd : data;
    size_t n = nd ? nlen : size;

    tsimd_added_seam_list_t seams;
    tsimd_tokenizer_added_seams(&tok, buf, n, &seams);

    ref_n = 0;
    size_t pos = 0;
    for (size_t s = 0; s <= seams.count; s++) {
        size_t soff = s < seams.count ? seams.items[s].off : n;
        if (soff > n) soff = n;
        if (soff < pos) soff = pos;
        if (soff > pos) {
            const uint8_t *gap = buf + pos;
            size_t glen = soff - pos;
            if (spm_mode || uni_mode) {
                if (uni_mode) {
                    uint8_t mbuf[2048];
                    int dummy = pos == 0 ? tok.spm_dummy_prefix : 0;
                    size_t mn = tsimd_metaspace_ex(gap, glen, mbuf,
                                                   sizeof mbuf, dummy);
                    if (mn >= sizeof mbuf) mn = sizeof mbuf - 1;
                    naive_gem4_span(mbuf, (uint32_t)mn);
                } else {
                    uint8_t cbuf[2048];
                    int dummy = pos == 0 ? tok.spm_dummy_prefix : 0;
                    size_t cn = tsimd_spm_normalize(gap, glen, cbuf,
                                                    sizeof cbuf - 1, dummy,
                                                    tok.spm_remove_extra);
                    if (cn == (size_t)-1) cn = 0;
                    if (cn >= sizeof cbuf) cn = sizeof cbuf - 1;
                    if (tok.vocab.scores)
                        naive_spm_span(cbuf, (uint32_t)cn);
                    else
                        naive_gem4_span(cbuf, (uint32_t)cn);
                }
            } else {
                tsimd_span_array_t sp;
                tsimd_span_array_init(&sp);
                if (tsimd_pretokenize(sch, gap, glen, &sp) != 0) {
                    tsimd_span_array_free(&sp);
                    free(nd);
                    return 0;
                }
                {
                    size_t before = ref_n;
                    for (size_t i = 0; i < sp.count; i++)
                        naive_bpe_span(gap + sp.spans[i].start,
                                       sp.spans[i].end - sp.spans[i].start);
                    if (getenv("TSIMD_FUZZ_DBG")) {
                        fprintf(stderr, "glen=%zu spans=%zu:", glen,
                                sp.count);
                        for (size_t i = 0; i < sp.count; i++)
                            fprintf(stderr, " [%u,%u)", sp.spans[i].start,
                                    sp.spans[i].end);
                        fprintf(stderr, "\nref piece:");
                        for (size_t i = before; i < ref_n; i++)
                            fprintf(stderr, " %u", ref_ids[i]);
                        fprintf(stderr, "\n");
                    }
                }
                tsimd_span_array_free(&sp);
            }
        }
        if (s < seams.count && ref_n < 4096)
            ref_ids[ref_n++] = tok.added[seams.items[s].idx].id;
        pos = soff + (s < seams.count ? seams.items[s].len : 0);
    }
    tsimd_added_seam_list_free(&seams);
    free(nd);

    if (mine_n != ref_n || memcmp(mine, ref_ids, mine_n * 4) != 0) {
        float mf, rf;
        if (mine_n != ref_n || !tok.vocab.scores) goto mismatch;
        for (size_t i = 0; i < mine_n; i++)
            if (mine[i] >= tok.vocab.vocab_size) goto mismatch;
        for (size_t i = 0; i < ref_n; i++)
            if (ref_ids[i] >= tok.vocab.vocab_size) goto mismatch;
        mf = tsimd_spm_path_score(&tok.vocab, mine, mine_n);
        rf = tsimd_spm_path_score(&tok.vocab, ref_ids, ref_n);
        if (tsimd_spm_accepts_reference_score(&tok.vocab, mine, mine_n,
                                              ref_ids, ref_n, mf, rf)) {
            if (getenv("TSIMD_FUZZ_DBG"))
                fprintf(stderr, "TIE sum mf=%g rf=%g\n", (double)mf,
                        (double)rf);
            return 0;
        }
mismatch:
        fprintf(stderr, "MISMATCH size=%zu mine=%zu ref=%zu\n", size,
                mine_n, ref_n);
        fprintf(stderr, "mine:");
        for (size_t i = 0; i < mine_n; i++) fprintf(stderr, " %u", mine[i]);
        fprintf(stderr, "\nref :");
        for (size_t i = 0; i < ref_n; i++) fprintf(stderr, " %u", ref_ids[i]);
        fprintf(stderr, "\n");
        abort();
    }
    return 0;
}
