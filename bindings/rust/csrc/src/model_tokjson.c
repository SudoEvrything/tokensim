#include "tokenizer.h"
#include "vocabulary.h"
#include "allocator.h"
#include "json.h"
#include "byte_unicode.h"
#include "input.h"
#include "model_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const TSIMD_PAT_GPT2 =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|"
    "\\s+(?!\\S)|\\s+";
static const char *const TSIMD_PAT_CL100K =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
    "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|"
    "\\s+(?!\\S)|\\s+";
static const char *const TSIMD_PAT_QWEN =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|"
    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
static const char *const TSIMD_PAT_O200K =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|"
    "\\s+(?!\\S)|\\s+";
static const char *const TSIMD_PAT_MINIMAX =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|"
    "\\s+(?!\\S)|\\s+";
static const char *const TSIMD_PAT_NEMO =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|"
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|"
    "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
static const char *const TSIMD_PAT_CJK =
    "[\344\270\200-\351\276\245\343\201\200-\343\202\237\343\202\240-\343\203\277]+";
static const char *const TSIMD_PAT_PUNCT =
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\015\012\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\015\012]*|\\s*[\015\012]+|\\s+(?!\\S)|\\s+";

static void collect_regex_patterns(const tsimd_json_value_t *v,
                                   const char **out, int *n) {
    if (!v || *n > 7) return;
    if (v->type == TSIMD_JSON_OBJECT) {
        if (v->u.obj.n == 0) return;
        for (size_t i = 0; i < v->u.obj.n; i++) {
            if (strcmp(v->u.obj.keys[i], "Regex") == 0) {
                const tsimd_json_value_t *val = v->u.obj.vals[i];
                if (val->type == TSIMD_JSON_STRING) {
                    if (*n <= 7) out[(*n)++] = val->u.str.ptr;
                }
            }
            collect_regex_patterns(v->u.obj.vals[i], out, n);
        }
    } else if (v->type == TSIMD_JSON_ARRAY) {
        if (v->u.arr.n == 0) return;
        for (size_t i = 0; i < v->u.arr.n; i++)
            collect_regex_patterns(v->u.arr.items[i], out, n);
    }
}

static int resolve_merge_pair(tsimd_vocab_t *v, const char *l, const char *r,
                              uint32_t slen_l, uint32_t slen_r, uint32_t rank,
                              const char *path, int raw_notation) {
    uint8_t lb[1024], rb[1024];
    uint32_t llen, rlen;

    if (raw_notation) {
        size_t nl = slen_l, nr = slen_r;
        if (nl > sizeof(lb) || nr > sizeof(rb)) {
            fprintf(stderr, "tsimd: %s merge %u side too long\n", path,
                    rank + 1);
            return -1;
        }
        memcpy(lb, l, nl);
        memcpy(rb, r, nr);
        llen = (uint32_t)nl;
        rlen = (uint32_t)nr;
    } else {
        if (tsimd_decode_remapped(l, slen_l, lb, sizeof(lb), &llen) != 0 ||
            tsimd_decode_remapped(r, slen_r, rb, sizeof(rb), &rlen) != 0) {
            fprintf(stderr, "tsimd: %s merge %u outside byte alphabet\n",
                    path, rank + 1);
            return -1;
        }
    }
    uint32_t lid = (raw_notation && llen == 1) ? v->byte_ids[lb[0]]
                                               : tsimd_vocab_find(v, lb, llen);
    uint32_t rid = (raw_notation && rlen == 1) ? v->byte_ids[rb[0]]
                                               : tsimd_vocab_find(v, rb, rlen);
    if (lid == UINT32_MAX || rid == UINT32_MAX) {
        fprintf(stderr,
                "tsimd: %s merge %u side not in vocabulary (l %.*s/%u r %.*s/%u)\n",
                path, rank + 1, llen, lb, llen, rlen, rb, rlen);
        return -1;
    }
    uint32_t m = tsimd_vocab_merge_target(v, lid, rid);
    if (m == UINT32_MAX) {
        fprintf(stderr,
                "tsimd: %s merge %u target not in vocabulary (l %u/%u r %u/%u)\n",
                path, rank + 1, lid, llen, rid, rlen);
        return -1;
    }
    if (tsimd_vocab_record_merge(v, lid, rid, m) != 0) return -1;
    tsimd_vocab_set_rank(v, m, rank);
    return 0;
}

static int has_digits_node(const tsimd_json_value_t *v) {
    if (!v) return 0;
    if (v->type == TSIMD_JSON_OBJECT) {
        const tsimd_json_value_t *t = tsimd_json_obj_get(v, "type");
        if (t && t->type == TSIMD_JSON_STRING &&
            strcmp(t->u.str.ptr, "Digits") == 0) {
            const tsimd_json_value_t *ind =
                tsimd_json_obj_get(v, "individual_digits");
            if (ind && ind->type == TSIMD_JSON_BOOL &&
                ind->u.boolean == 1)
                return 1;
        }
        for (size_t i = 0; i < v->u.obj.n; i++)
            if (has_digits_node(v->u.obj.vals[i])) return 1;
        return 0;
    }
    if (v->type == TSIMD_JSON_ARRAY) {
        if (v->u.arr.n == 0) return 0;
        for (size_t i = 0; i < v->u.arr.n; i++)
            if (has_digits_node(v->u.arr.items[i])) return 1;
    }
    return 0;
}

static int detect_bytelevel(const tsimd_json_value_t *v, int *depth) {
    if (!v) return 0;
    if (*depth > 16) return 0;
    if (++*depth == 17) return 0;
    if (v->type == TSIMD_JSON_OBJECT) {
        const tsimd_json_value_t *t = tsimd_json_obj_get(v, "type");
        if (t && t->type == TSIMD_JSON_STRING &&
            strcmp(t->u.str.ptr, "ByteLevel") == 0)
            return 1;
        for (size_t i = 0; i < v->u.obj.n; i++)
            if (detect_bytelevel(v->u.obj.vals[i], depth)) return 1;
        return 0;
    }
    if (v->type == TSIMD_JSON_ARRAY) {
        if (v->u.arr.n == 0) return 0;
        for (size_t i = 0; i < v->u.arr.n; i++)
            if (detect_bytelevel(v->u.arr.items[i], depth)) return 1;
    }
    return 0;
}

static int load_added_tokens(const tsimd_json_value_t *root,
                             tsimd_added_token_t **added_out,
                             size_t *n_added_out, char **blob_out) {
    const tsimd_json_value_t *arr =
        tsimd_json_obj_get(root, "added_tokens");
    if (!arr) {
        *added_out = NULL;
        *n_added_out = 0;
        *blob_out = NULL;
        return 0;
    }
    if (arr->type != TSIMD_JSON_ARRAY) {
        fprintf(stderr, "tsimd: malformed added_tokens\n");
        return -1;
    }
    size_t n = arr->u.arr.n;
    size_t blob_bytes = 0;
    for (size_t i = 0; i < n; i++) {
        const tsimd_json_value_t *e = arr->u.arr.items[i];
        if (!e || e->type != TSIMD_JSON_OBJECT) {
            fprintf(stderr, "tsimd: malformed added_tokens\n");
            return -1;
        }
        const tsimd_json_value_t *c = tsimd_json_obj_get(e, "content");
        if (!c || c->type != TSIMD_JSON_STRING) {
            fprintf(stderr, "tsimd: malformed added_tokens\n");
            return -1;
        }
        blob_bytes += c->u.str.len + 1;
    }
    tsimd_added_token_t *out = NULL;
    char *blob = NULL;
    if (n > 0) {
        out = malloc(n * sizeof(*out));
        blob = malloc(blob_bytes);
        if (!out || !blob) {
            free(out);
            free(blob);
            fprintf(stderr, "tsimd: malformed added_tokens\n");
            return -1;
        }
    }
    size_t off = 0;
    for (size_t i = 0; i < n; i++) {
        const tsimd_json_value_t *e = arr->u.arr.items[i];
        const tsimd_json_value_t *c = tsimd_json_obj_get(e, "content");
        const tsimd_json_value_t *idv = tsimd_json_obj_get(e, "id");
        const tsimd_json_value_t *sp = tsimd_json_obj_get(e, "special");
        memcpy(blob + off, c->u.str.ptr, c->u.str.len);
        blob[off + c->u.str.len] = 0;
        out[i].text = blob + off;
        out[i].length = c->u.str.len;
        out[i].id = idv ? (uint32_t)tsimd_json_int(idv) : 0;
        out[i].is_special =
            (sp && sp->type == TSIMD_JSON_BOOL && sp->u.boolean) ? 1 : 0;
        off += c->u.str.len + 1;
    }
    *added_out = out;
    *n_added_out = n;
    *blob_out = blob;
    return 0;
}

static int classify_pattern(const char *p) {
    if (strcmp(p, TSIMD_PAT_MINIMAX) == 0) return TSIMD_PRETOK_O200K;
    if (strcmp(p, TSIMD_PAT_GPT2) == 0) return TSIMD_PRETOK_GPT2;
    if (strcmp(p, TSIMD_PAT_CL100K) == 0) return TSIMD_PRETOK_CL100K;
    if (strcmp(p, TSIMD_PAT_QWEN) == 0) return TSIMD_PRETOK_QWEN25;
    if (strcmp(p, TSIMD_PAT_O200K) == 0) return TSIMD_PRETOK_O200K;
    if (strcmp(p, TSIMD_PAT_NEMO) == 0) return TSIMD_PRETOK_NEMO;
    return -1;
}

int tsimd_model_load_tokenizer_json(tsimd_vocab_t *v, const char *path,
                                    tsimd_pretok_scheme_t *scheme,
                                    tsimd_added_token_t **added_out,
                                    size_t *n_added_out,
                                    char **added_blob_out,
                                    int *norm_out) {
    tsimd_byteunicode_init();
    *added_out = NULL;
    *n_added_out = 0;
    *added_blob_out = NULL;
    *norm_out = 0;

    tsimd_input_t in;
    if (tsimd_input_read_file(path, &in) != 0) {
        fprintf(stderr, "tsimd: cannot read %s\n", path);
        return -1;
    }
    tsimd_arena_t arena;
    tsimd_arena_init(&arena);
    size_t err_off = 0;
    const tsimd_json_value_t *root =
        tsimd_json_parse(&arena, in.data, in.length, &err_off);
    if (!root || root->type != TSIMD_JSON_OBJECT) {
        fprintf(stderr, "tsimd: %s:%zu invalid JSON\n", path, err_off);
        tsimd_arena_reset(&arena);
        tsimd_input_release(&in);
        return -1;
    }

    const tsimd_json_value_t *norm_node =
        tsimd_json_obj_get(root, "normalizer");
    int norm_val = 0;
    int gemma_norm = 0;
    if (norm_node && norm_node->type != TSIMD_JSON_NULL) {
        const tsimd_json_value_t *tv = tsimd_json_obj_get(norm_node, "type");
        const tsimd_json_value_t *seq =
            tsimd_json_obj_get(norm_node, "normalizers");
        if (!tv || tv->type != TSIMD_JSON_STRING) {
            fprintf(stderr,
                    "tsimd: %s has a normalizer; only null, empty Sequence, "
                    "Gemma Replace, NFC and NFKC are supported\n",
                    path);
            tsimd_arena_reset(&arena);
            tsimd_input_release(&in);
            return -1;
        }
        int seq_empty = 0;
        if (seq && seq->type == TSIMD_JSON_ARRAY &&
            strcmp(tv->u.str.ptr, "Sequence") == 0)
            seq_empty = (seq->u.arr.n == 0);
        if (strcmp(tv->u.str.ptr, "Replace") == 0) {
            const tsimd_json_value_t *pat =
                tsimd_json_obj_get(norm_node, "pattern");
            const tsimd_json_value_t *strv =
                pat ? tsimd_json_obj_get(pat, "String") : NULL;
            const tsimd_json_value_t *content =
                tsimd_json_obj_get(norm_node, "content");
            if (strv && strv->type == TSIMD_JSON_STRING) {
                const char *s = strv->u.str.ptr;
                int ok = (s[0] == ' ') ? (s[1] == 0) : 0;
                gemma_norm = (ok && content != NULL) ? 1 : 0;
                if (gemma_norm) goto norm_done;
            }
        }
        if (seq_empty) goto norm_done;
        if (strcmp(tv->u.str.ptr, "NFC") == 0) {
            gemma_norm = 0;
            norm_val = TSIMD_NORM_NFC;
            goto norm_done;
        }
        if (strcmp(tv->u.str.ptr, "NFKC") == 0) {
            gemma_norm = 0;
            norm_val = TSIMD_NORM_NFKC;
            goto norm_done;
        }
        fprintf(stderr,
                "tsimd: %s has a normalizer; only null, empty Sequence, "
                "Gemma Replace, NFC and NFKC are supported\n",
                path);
        tsimd_arena_reset(&arena);
        tsimd_input_release(&in);
        return -1;
    }
norm_done:

    const tsimd_json_value_t *model = tsimd_json_obj_get(root, "model");
    if (!model) {
        fprintf(stderr, "tsimd: %s is not a BPE tokenizer.json\n", path);
        tsimd_arena_reset(&arena);
        tsimd_input_release(&in);
        return -1;
    }
    const tsimd_json_value_t *mtv = tsimd_json_obj_get(model, "type");
    if (!mtv || mtv->type != TSIMD_JSON_STRING ||
        strcmp(mtv->u.str.ptr, "BPE") != 0) {
        fprintf(stderr, "tsimd: %s is not a BPE tokenizer.json\n", path);
        tsimd_arena_reset(&arena);
        tsimd_input_release(&in);
        return -1;
    }

    const tsimd_json_value_t *pt =
        tsimd_json_obj_get(root, "pre_tokenizer");
    if (!pt) {
        fprintf(stderr, "tsimd: tokenizer.json has no pre_tokenizer\n");
        tsimd_arena_reset(&arena);
        tsimd_input_release(&in);
        return -1;
    }

    uint32_t byte_ids[256];
    uint32_t byte_ids_hex[256];
    int byte_seen[256];
    int depth = 0;
    int has_bl = detect_bytelevel(pt, &depth);
    int det = -1;
    int gemma_pt = 0;
    const char *pats[8];
    int npats = 0;

    if (pt->type == TSIMD_JSON_OBJECT) {
        const tsimd_json_value_t *tv = tsimd_json_obj_get(pt, "type");
        if (tv && tv->type == TSIMD_JSON_STRING &&
            strcmp(tv->u.str.ptr, "Sequence") == 0) {
            const tsimd_json_value_t *ps =
                tsimd_json_obj_get(pt, "pretokenizers");
            if (ps && ps->type == TSIMD_JSON_ARRAY && ps->u.arr.n == 4) {
                const char *want[3] = {"\\p{N}{1,3}", TSIMD_PAT_CJK,
                                       TSIMD_PAT_PUNCT};
                int ok = 1;
                for (int i = 0; i < 3 && ok; i++) {
                    const tsimd_json_value_t *item = ps->u.arr.items[i];
                    if (!item || item->type != TSIMD_JSON_OBJECT) {
                        ok = 0;
                        break;
                    }
                    const tsimd_json_value_t *itv =
                        tsimd_json_obj_get(item, "type");
                    if (!itv || itv->type != TSIMD_JSON_STRING ||
                        strcmp(itv->u.str.ptr, "Split") != 0) {
                        ok = 0;
                        break;
                    }
                    const tsimd_json_value_t *pv =
                        tsimd_json_obj_get(item, "pattern");
                    if (!pv) {
                        ok = 0;
                        break;
                    }
                    const tsimd_json_value_t *rv =
                        tsimd_json_obj_get(pv, "Regex");
                    if (!rv || rv->type != TSIMD_JSON_STRING ||
                        strcmp(rv->u.str.ptr, want[i]) != 0) {
                        ok = 0;
                        break;
                    }
                }
                if (ok) {
                    const tsimd_json_value_t *bl = ps->u.arr.items[3];
                    if (bl) {
                        const tsimd_json_value_t *bt =
                            tsimd_json_obj_get(bl, "type");
                        if (bt && bt->type == TSIMD_JSON_STRING &&
                            strcmp(bt->u.str.ptr, "ByteLevel") == 0) {
                            const tsimd_json_value_t *ur =
                                tsimd_json_obj_get(bl, "use_regex");
                            if (!(ur && ur->type == TSIMD_JSON_BOOL &&
                                  ur->u.boolean))
                                det = TSIMD_PRETOK_DEEPSEEK;
                        }
                    }
                }
            }
        }
        if (det < 0) {
            const tsimd_json_value_t *tv2 = tsimd_json_obj_get(pt, "type");
            if (tv2 && tv2->type == TSIMD_JSON_STRING &&
                strcmp(tv2->u.str.ptr, "Split") == 0) {
                const tsimd_json_value_t *pv =
                    tsimd_json_obj_get(pt, "pattern");
                if (pv) {
                    const tsimd_json_value_t *sv =
                        tsimd_json_obj_get(pv, "String");
                    if (sv && sv->type == TSIMD_JSON_STRING &&
                        sv->u.str.ptr[0] == ' ' &&
                        sv->u.str.ptr[1] == 0) {
                        const tsimd_json_value_t *inv =
                            tsimd_json_obj_get(pt, "invert");
                        if (!(inv && inv->type == TSIMD_JSON_BOOL &&
                              inv->u.boolean)) {
                            gemma_pt = 1;
                            det = TSIMD_PRETOK_GEM4;
                        }
                    }
                }
            }
        }
        if (det < 0) collect_regex_patterns(pt, pats, &npats);
    } else {
        collect_regex_patterns(pt, pats, &npats);
    }

    if (det < 0 && npats > 0) {
        for (int i = 0; i < npats; i++) {
            int s = classify_pattern(pats[i]);
            if (s < 0) {
                fprintf(stderr,
                        "tsimd: unsupported pre_tokenizer pattern: \"%s\"\n",
                        pats[i]);
                tsimd_arena_reset(&arena);
                tsimd_input_release(&in);
                return -1;
            }
            if (det >= 0 && det != s) {
                fwrite("tsimd: pre_tokenizer mixes incompatible patterns\n",
                       1, 49, stderr);
                tsimd_arena_reset(&arena);
                tsimd_input_release(&in);
                return -1;
            }
            det = s;
        }
    }
    if (det < 0 && npats == 0) {
        if (!has_bl) {
            fprintf(stderr, "tsimd: unsupported pre_tokenizer (no known "
                            "pattern)\n");
            tsimd_arena_reset(&arena);
            tsimd_input_release(&in);
            return -1;
        }
        {
            const tsimd_json_value_t *ur =
                tsimd_json_obj_get(pt, "use_regex");
            if (ur && ur->type == TSIMD_JSON_BOOL && !ur->u.boolean) {
                fprintf(stderr,
                        "tsimd: ByteLevel pre_tokenizer with use_regex=false "
                        "is unsupported\n");
                tsimd_arena_reset(&arena);
                tsimd_input_release(&in);
                return -1;
            }
        }
        det = has_digits_node(pt) ? TSIMD_PRETOK_GPT2D
                                     : TSIMD_PRETOK_GPT2;
    }

    if (gemma_pt != gemma_norm) {
        fprintf(stderr,
                "tsimd: %s Gemma metaspace normalizer and pre_tokenizer must "
                "appear together\n",
                path);
        tsimd_arena_reset(&arena);
        tsimd_input_release(&in);
        return -1;
    }

    *scheme = (tsimd_pretok_scheme_t)det;
    *norm_out = norm_val;

    if (tsimd_vocab_build_start(v) != 0) {
        tsimd_arena_reset(&arena);
        tsimd_input_release(&in);
        return -1;
    }

    const tsimd_json_value_t *vocab = tsimd_json_obj_get(model, "vocab");
    if (!vocab || vocab->type != TSIMD_JSON_OBJECT) {
        fprintf(stderr, "tsimd: %s model has no vocab\n", path);
        goto fail;
    }

    {
        memset(byte_ids, 0xFF, sizeof(byte_ids));
        {
            const tsimd_json_value_t *bf =
                tsimd_json_obj_get(model, "byte_fallback");
            if (bf && bf->type == TSIMD_JSON_BOOL && bf->u.boolean)
                v->byte_fallback = 1;
        }
        memset(byte_ids_hex, 0xFF, sizeof(byte_ids_hex));
        memset(byte_seen, 0, sizeof(byte_seen));
        uint32_t next_id = 0;
        size_t buf_cap = 4096;
        uint8_t *dec = malloc(buf_cap);
        if (!dec) {
            fprintf(stderr, "tsimd: %s out of memory\n", path);
            goto fail;
        }

        for (size_t i = 0; i < vocab->u.obj.n; i++) {
            const char *key = vocab->u.obj.keys[i];
            long long idv = tsimd_json_int(vocab->u.obj.vals[i]);
            uint32_t id = (uint32_t)idv;
            if ((long long)id != idv || id == UINT32_MAX) {
                fprintf(stderr, "tsimd: %s invalid vocab entry\n", path);
                free(dec);
                goto fail;
            }
            size_t klen = vocab->u.obj.klens[i];
            const uint8_t *addp;
            uint32_t alen;
            uint8_t hexb;
            if (det == TSIMD_PRETOK_GEM4) {
                addp = (const uint8_t *)key;
                alen = (uint32_t)klen;
                if (klen == 6 && key[0] == '<' && key[1] == '0' &&
                    key[2] == 'x' && key[5] == '>') {
                    unsigned hv[2];
                    int ok = 1;
                    for (int q = 0; q < 2; q++) {
                        uint8_t c = (uint8_t)key[3 + q];
                        if (c >= '0' && c <= '9') hv[q] = c - '0';
                        else if (c >= 'a' && c <= 'f') hv[q] = c - 'a' + 10;
                        else if (c >= 'A' && c <= 'F') hv[q] = c - 'A' + 10;
                        else hv[q] = 0;
                        (void)ok;
                    }
                    hexb = (uint8_t)((hv[0] << 4) | hv[1]);
                    addp = &hexb;
                    alen = 1;
                    byte_ids_hex[hexb] = id;
                    if (byte_ids[hexb] == 0xFFFFFFFFu) byte_ids[hexb] = id;
                    byte_seen[hexb] = 1;
                }
                if (klen == 1) byte_ids[addp[0]] = id;
                uint8_t one[16];
                uint32_t dlen = 0;
                if (tsimd_decode_remapped(key, klen, one, sizeof(one),
                                          &dlen) == 0 &&
                    dlen == 1)
                    byte_seen[one[0]] = 1;
            } else {
                if (klen + 1 > buf_cap) {
                    buf_cap = klen + 64;
                    uint8_t *grown = realloc(dec, buf_cap);
                    if (!grown) {
                        fprintf(stderr, "tsimd: %s out of memory\n", path);
                        free(dec);
                        goto fail;
                    }
                    dec = grown;
                }
                uint32_t dlen = 0;
                if (tsimd_decode_remapped(key, klen, dec, buf_cap,
                                          &dlen) == 0) {
                    addp = dec;
                    alen = dlen;
                    if (alen == 1) {
                        byte_ids[addp[0]] = id;
                        byte_seen[addp[0]] = 1;
                    }
                } else {
                    addp = (const uint8_t *)key;
                    alen = (uint32_t)klen;
                }
            }
            if (tsimd_vocab_add_token(v, addp, alen, id) == UINT32_MAX) {
                fprintf(stderr, "tsimd: %s invalid vocab entry\n", path);
                free(dec);
                goto fail;
            }
            if (id + 1 > next_id) next_id = id + 1;
        }

        for (int b = 0; b < 256; b++) {
            if (byte_seen[b]) continue;
            uint8_t one = (uint8_t)b;
            if (tsimd_vocab_add_token(v, &one, 1, next_id) == UINT32_MAX) {
                fprintf(stderr, "tsimd: %s byte-fallback add failed\n", path);
                free(dec);
                goto fail;
            }
            if (byte_ids[b] == 0xFFFFFFFFu) byte_ids[b] = next_id;
            if (byte_ids_hex[b] == 0xFFFFFFFFu) byte_ids_hex[b] = next_id;
            next_id++;
            byte_seen[b] = 1;
        }
        free(dec);
        if (det != TSIMD_PRETOK_GEM4) {
            for (int b = 0; b < 256; b++) {
                if (!byte_seen[b]) {
                    fprintf(stderr,
                            "tsimd: %s vocabulary is not byte-complete\n",
                            path);
                    goto fail;
                }
            }
        }
    }

    {
        const tsimd_json_value_t *merges =
            tsimd_json_obj_get(model, "merges");
        if (!merges || merges->type != TSIMD_JSON_ARRAY) {
            fprintf(stderr, "tsimd: %s model has no merges\n", path);
            goto fail;
        }
        if (tsimd_vocab_seal(v, merges->u.arr.n) != 0) goto fail;
    for (int b = 0; b < 256; b++) {
        int ctrl = (b < 0x20 && b != 0x09 && b != 0x0a && b != 0x0d) ||
                   b == 0x7f;
        v->raw_piece_units = (det == TSIMD_PRETOK_GEM4);
        v->ignore_merges = 0;
        {
            const tsimd_json_value_t *im =
                tsimd_json_obj_get(model, "ignore_merges");
            if (im && im->type == TSIMD_JSON_BOOL && im->u.boolean)
                v->ignore_merges = 1;
        }
        if (det == TSIMD_PRETOK_GEM4) {
            if (byte_ids[b] != 0xFFFFFFFFu)
                v->byte_ids[b] = byte_ids[b];
            else if (byte_ids_hex[b] != 0xFFFFFFFFu)
                v->byte_ids[b] = byte_ids_hex[b];
        } else if (ctrl && byte_ids_hex[b] != 0xFFFFFFFFu)
            v->byte_ids[b] = byte_ids_hex[b];
        else if (byte_ids[b] != 0xFFFFFFFFu)
            v->byte_ids[b] = byte_ids[b];
        else if (byte_ids_hex[b] != 0xFFFFFFFFu)
            v->byte_ids[b] = byte_ids_hex[b];
        v->byte_ids_hex[b] = byte_ids_hex[b];
    }

        for (size_t i = 0; i < merges->u.arr.n; i++) {
            const tsimd_json_value_t *m = merges->u.arr.items[i];
            if (m->type == TSIMD_JSON_ARRAY && m->u.arr.n == 2 &&
                m->u.arr.items[0]->type == TSIMD_JSON_STRING &&
                m->u.arr.items[1]->type == TSIMD_JSON_STRING) {
                if (resolve_merge_pair(
                        v, m->u.arr.items[0]->u.str.ptr,
                        m->u.arr.items[1]->u.str.ptr,
                        m->u.arr.items[0]->u.str.len,
                        m->u.arr.items[1]->u.str.len, (uint32_t)i, path,
                        det == TSIMD_PRETOK_GEM4) != 0)
                    goto fail;
            } else if (m->type == TSIMD_JSON_STRING) {
                const char *s = m->u.str.ptr;
                const char *sp = strchr(s, ' ');
                if (!sp) {
                    fprintf(stderr, "tsimd: %s malformed merge string\n",
                            path);
                    goto fail;
                }
                char l[512];
                size_t ll = (size_t)(sp - s);
                size_t rl = m->u.str.len - ll - 1;
                if (ll >= sizeof(l)) {
                    fprintf(stderr, "tsimd: %s merge too long\n", path);
                    goto fail;
                }
                memcpy(l, s, ll);
                l[ll] = 0;
                if (resolve_merge_pair(v, l, sp + 1, (uint32_t)ll,
                                       (uint32_t)rl, (uint32_t)i, path,
                                       det == TSIMD_PRETOK_GEM4) != 0)
                    goto fail;
            } else {
                fprintf(stderr, "tsimd: %s malformed merge entry\n", path);
                goto fail;
            }
        }
    }

    if (load_added_tokens(root, added_out, n_added_out, added_blob_out) != 0)
        goto fail;

    if (tsimd_vocab_pair_rank_build(v) != 0) {
        fprintf(stderr, "tsimd: %s cannot build pair tables\n", path);
        goto fail;
    }
    if (tsimd_vocab_sim_build(v) != 0) {
        fprintf(stderr, "tsimd: %s cannot build merge simulation\n", path);
        goto fail;
    }

    if (det == TSIMD_PRETOK_GEM4) {
        (void)tsimd_vocab_spm_cross_ready(v);
        (void)tsimd_vocab_spm_seam_ok(v);
    }

    tsimd_arena_reset(&arena);
    tsimd_input_release(&in);
    return 0;

fail:
    tsimd_arena_reset(&arena);
    tsimd_input_release(&in);
    tsimd_vocab_free(v);
    free(*added_out);
    free(*added_blob_out);
    *added_out = NULL;
    *n_added_out = 0;
    *added_blob_out = NULL;
    return -1;
}
