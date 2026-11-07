The par-vs-seq divergences found by tests/fuzz_par_parity.c (run 362).

The class: the parallel windowed unigram/SPM path (encode_parallel on
the GEM4/SPM models) truncates unk runs on invalid-UTF-8 content. The
sequential path is the reference-exact one (full_corpus_parity). The
known-open windowed-DP incompleteness (runs 297-316) with the first
deterministic repros.

The repro (each file, the spec per family):
  clang -fsanitize=fuzzer -O2 -g -I include tests/fuzz_par_parity.c \
      build/libtokensimd.a -lm -lpthread -o /tmp/fp
  TSIMD_FUZZ_SPEC="models/reference/mistral7b_tokenizer.model;spm" \
      /tmp/fp tests/fixtures/par_unigram_divergence/mistral_par_trunc.bin
  -> PAR-COUNT th=3/6/16 par=285973 seq=786425 (the abort)

  gemma3-4b_par_half.bin: par=112824 seq=225648 (the GEM4 metaspace spec)
  gemma4_par_short.bin: the same class (the gemma4 spec)

## FIND D (run 363): the giant-word quadratic cliff

mistral_giant_word_cliff.bin (30 bytes: 0a 00 00 00 40 00 00 00 +
20x 0x7e + 00 00). Expanded to a single no-whitespace "word", the
merge-heavy content hits the SPM-BPE huge-unit path: 60 KB encodes
fine, 600 KB exceeds 60 s (the O(n^2)+ merge class). The
SEQUENTIAL path has the same cliff - the par fix (the wrong
normalizer) made the par share it. A DoS-class hazard for
untrusted input. The fix direction: the word-size cap / the
blocked-path complexity bound. The fuzz harness reproduces with
-timeout=8; the mistral par fuzz budget stays timeout-prone until
this lands.

## The kimi span finding (run 369, RESOLVED in run 371)

kimi_qs_span.bin: {0x11, 0x3f, 0x73}. The run-369 diagnosis was
WRONG: under the real kimi pattern (tokenization_kimi.py, the
tiktoken.Encoding reference) the '?' and the 's' are NOT separate -
tiktoken splits "\x11?s" as ["\x11?", "s"] (the punct run takes
0x11 AND '?', then the word "s"; "?s" alone = one span, token
48097). Our production encode route (the mask walker +
tsimd_match_kimi) produces exactly that - the encode = [205, 30,
82] = tiktoken EXACT.

The run-369 "masks glue ?s" observation came from
tsimd_pretokenize(TSIMD_PRETOK_KIMI, ...), which fell through its
dispatch chain to tsimd_pretokenize_deepseek - the deepseek walker
produced the [0,1) [1,3) shapes. Run 371 fixed the dispatch (the
kimi line added before the deepseek fall-through), added the
dispatch-equivalence check to test_kimi_masks (68 cases) and the
artifact cases CS("\x11?s") + the multi-shape case. The scalar
matcher, the mask walker and the dispatched API now agree on every
case; test_kimi_masks OK, test_kimi OK.

The naive fuzz oracle (merge-based, scheme-less) still disagrees
with tiktoken on this shape BY DESIGN (it runs no pretokenizer at
all) - the scheme-less kimi differential stays.
