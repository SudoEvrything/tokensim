#include "bpe.h"
#include "tokenizer.h"

#include <stdio.h>
#include <string.h>

static int fails = 0;

static void probe(const char *path, const char *name) {
    tsimd_tokenizer_t tok;
    memset(&tok, 0, sizeof tok);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&tok, TSIMD_MODEL_TOKENIZER_JSON, path,
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "FAIL load %s\n", name);
        fails++;
        return;
    }
    const tsimd_vocab_t *v = &tok.vocab;
    unsigned reach_n = 0, sim_n = 0, reach_seed = 0, sim_seed = 0;
    for (uint32_t id = 0; id < v->vocab_size; id++) {
        uint32_t len = v->offsets[id + 1] - v->offsets[id];
        if (len < 1 || len > TSIMD_PCACHE_MAX_SPAN) continue;
        const uint8_t *b = v->bytes + v->offsets[id];
        if (len == 1 && v->byte_ids[b[0]] != id &&
            v->byte_ids[b[0]] < v->vocab_size)
            continue;
        uint32_t z = 0;
        while (z < len && z < 8 && b[z] == 0) z++;
        if (z == 8) continue;
        int gated = v->pair_table ? 1 : 0;
        int ok_reach = !gated || !v->merge_reach || v->merge_reach[id];
        int ok_sim = !gated || !v->merge_sim || v->merge_sim[id];
        if (ok_reach) reach_seed++;
        if (ok_sim) sim_seed++;
        if (v->merge_reach && v->merge_reach[id]) reach_n++;
        if (v->merge_sim && v->merge_sim[id]) sim_n++;
    }
    printf(
        "%s: vocab=%u reach_ok=%u sim_ok=%u seed_reach=%u seed_sim=%u "
        "delta=%d\n",
        name, v->vocab_size, reach_n, sim_n, reach_seed, sim_seed,
        (int)reach_seed - (int)sim_seed);
    tsimd_tokenizer_free(&tok);
}

int main(void) {
    probe(TSIMD_SRC_DIR "/models/reference/gemma4_tokenizer.json",
          "gemma4");
    probe(TSIMD_SRC_DIR "/models/reference/deepseek_tokenizer.json",
          "deepseek");
    probe(TSIMD_SRC_DIR "/models/reference/smollm2_tokenizer.json",
          "smollm2");
    probe(TSIMD_SRC_DIR "/models/reference/qwen2.5-0.5b_tokenizer.json",
          "qwen2.5");
    return fails != 0;
}
