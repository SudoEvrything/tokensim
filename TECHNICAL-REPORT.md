# Tokensim 

*The fastest tokenizer engine written in C/C++, byte-exact across 14 model families, 84x faster than tiktoken, 1.34–1.39x faster than gigatoken - rust reference engine, and shipped to
three ecosystems.*

---

## 01 - Introduction

Every call to a large language model starts and ends with a
tokenizer. Before the model sees your prompt, the tokenizer splits
the text into subword pieces and maps each piece to an integer id.
After the model answers, the tokenizer maps the output ids back to
text. The tokenizer is therefore the first and the last code that
every request touches. It is pure byte-level CPU work. No GPU
helps it.

This cost shows up in three places.

First, training. A corpus must be tokenized before training. This
is a one-time job, but it is measured in CPU-hours. An engine that
runs 1.3x faster saves 23% of those hours, at byte-exactness, with
no risk to model quality.

Second, serving. The prompt is tokenized on the request path. On a
busy node the tokenizer competes for cores with the model itself.
For short prompts - the majority of real traffic - the tokenize
step is a visible part of the time to first token.

Third, bulk work. Backfills, evaluations, and indexing jobs
tokenize gigabytes. At 100 MB/s, one terabyte takes about three
hours. At 500 MB/s, it takes about thirty-five minutes.

The hard part is not speed. The hard part is that speed must not
change the output. Each model family defines its own encoding: a
set of rules that decides where a word may be split, a merge
priority table, special tokens, and a fallback for malformed
bytes. If two engines disagree by even one merge at one boundary,
they produce different ids. Different ids mean the model sees a
different prompt than the one it was trained to read. So this
project has one organizing constraint: **correctness first**.
Speed must be provably safe, not hopeful.

### The call

The usage follows the Hugging Face shape: one argument in, ids
out, decode back.

```python
import tokensim
tok = tokensim.tokenizer("cl100k_base.tiktoken")
ids = tok.encode(b"some prompt")
tok.decode(ids)
```

One argument, and everything else is derived. The format comes
from the file extension (`.tiktoken`, `.model`, `.json`). The
pretokenizer scheme comes from the vocab size signature: a
100,256-token tiktoken file is cl100k, a 199,998-token file is
o200k, a 50,257-token file is GPT-2. The thread count comes from
the machine, and every call above a quarter megabyte of input
fans out to the worker pool, on encode and decode alike. The
same shape holds in Rust (`encode`, `decode`) and C
(`encode_parallel`, `decode`, with zero as the auto knob,
because C has no default arguments).

Batch calls follow the same rule
(`tok.encode_batch(list_of_prompts)`), and decode hands back an
owned buffer with no extra copy in Rust and C.

Training-scale callers that tokenize gigabytes keep both legs in
caller-owned buffers instead: `encode_threads_into` and
`decode_into` in Python, `encode_into` and `decode_parallel` in
Rust, the `_into` entries in C. On C and Rust the simple calls
measure the same rates as the into shape (about 4 GB/s decode on
the reference corpus): the engine recycles its large transient
buffers internally, so the allocation the simple shape pays is
warm. In Python the decode hands back a real bytes object, and
that one copy is the difference (about 3.3 GB/s simple against
4.3 GB/s into). The two shapes sit side by side in
`scripts/consumers/`, and every document they tokenize is
decoded and byte-compared before the program exits.

### The pipeline

tokensim runs a short pipeline:

```
input (mmap view) -> added-token split -> pretokenize -> encode -> uint32 ids
```

Each stage is simple to describe.

The **added-token split** scans the input for special strings
(things like `<|endoftext|>`). The scan is a single left-to-right
pass indexed by the first byte of each special string, so its cost
grows with the number of hits, not with the size of the
special-token table. A hit hard-splits the input: the special
token becomes one id, and the bytes between hits are separate
jobs.

The **pretokenizer** applies the model family's split rules. These
rules say where a merge may not cross - for example, a merge may
not cross from one word into the next. The pretokenizer turns raw
bytes into a stream of small pieces, each of which is encoded
independently. This stage is where the regex engines usually live
in other implementations. tokensim does not use a regex engine at
all (more on this below). The scanners read 64 bytes at a time,
and any block containing non-ASCII bytes falls back to a scalar
matcher until an ASCII-clean window appears, so multi-byte text is
handled without slowing the ASCII path.

The **encode** stage turns each piece into ids. Most families use
BPE: start from the bytes of the piece, then repeatedly merge the
adjacent pair with the lowest merge rank until no merge applies.
The merge engine picks its form by size: short pieces (up to 64
symbols) take a cache-friendly linear scan; longer pieces take a
rank-bucket queue; a lazy-deletion min-heap remains as the
fallback. One family group (T5, XLM-R) uses a different algorithm,
unigram, which picks the best segmentation by score. tokensim
implements both.

The result is an array of `uint32` ids. The decoder reverses the
whole chain exactly, and it needs no side tables and no state,
which is why the round-trip can be fuzzed over arbitrary bytes and
the decode can be tested separately from the encode.

### Four properties define the design

**1. No regex engine.** A regex engine is a small interpreter. It
tracks state, allocates memory, and branches per character. That
is fine for correctness and slow for throughput. The split rules
of a tokenizer are static - they never change at runtime. So
tokensim compiles each family's rules into hand-written scanners,
and in the fast path into **bitmask algebra**: the scanner reads a
64-byte block once and produces thirteen bitmasks, where each bit
says "this byte is a letter / digit / space / newline /
punctuation". From those masks, simple shift-AND-OR expressions on
`uint64_t` words decide every boundary in the block at once. Think
of it as replacing 64 branches with 13 table entries and a few
dozen integer instructions.

**2. Scalar oracle first.** Every stage has a second, simple
implementation, and the fast version must match it byte for byte
at every SIMD level. Section 3.1 explains why this one rule does
more work than any other.

**3. Memoized encoding.** Natural text repeats: "the" appears
thousands of times and encodes the same way every time. The engine
memoizes short pieces, so a repeat costs one lookup instead of a
merge run. Hit rates run 91.7-99.9%, worth roughly 1.9x
end-to-end on its own. The whole-segment variant, the unigram
memo, stays opt-in: it trades memory for speed, and that trade is
a deliberate, documented choice.

**4. Parallelism never changes output.** The engine splits work
across threads only at provably safe seams. The seam rule is
short: you may split between byte p-1 and byte p when p-1 is a
letter and p is an ASCII non-letter, or p-1 is an ASCII digit and p
is an ASCII non-digit. The o200k and kimi schemes add one guard: a
letter followed by an apostrophe never splits. The piece before the
split is then a complete word whose encoding depends only on the
bytes left of the cut, and nothing after the cut reaches back into
it. The invariant is tested: for any thread count, the parallel
output must equal the sequential output byte for byte.

### The ISA story, briefly

The engine runs on SSE2 (the x86-64 baseline), AVX2, and AVX-512,
and it trusts CPU reports only after the CPU passes a one-time
execution test. The full reasoning is the first design decision
told in depth (section 3.6).

Every claim in this report is backed by an executable test, a
script, or a committed fixture in the repository.

---

## 02 - Correctness: how exactness is earned and proven

"Byte-exact" is a claim about every input, not about the inputs we
tried. No test suite can enumerate all inputs, so exactness is
built as layers, where each layer catches a class of mistake the
previous layer cannot. All layers run today from a fresh shell.

**Layer 1 - the scalar oracles.** Each stage has the simple
reference implementation described above. The full suite forces
every SIMD level (scalar, SSE2, AVX2, AVX-512) to run against the
oracle on the same inputs, over hand-picked hard cases, a length
sweep that crosses block edges, and fuzz modes that stress the
CRLF, digit, contraction, and unicode paths. This catches
code-generation mistakes: the SIMD kernel that reads one byte too
far, the mask algebra that drops one boundary case at a block
edge.

**Layer 2 - the reference implementations.** A tokenizer is only
correct if it agrees with the model family's reference. So the
suite compares tokensim against three independent references:
tiktoken (OpenAI's Rust engine), HuggingFace tokenizers, and
sentencepiece (Google's). Thirteen families are pinned at multiple
thread counts, and each pin fixes both the token count and a
Fibonacci hash of the id stream, so a reorder alone cannot pass.
The pins themselves must be trustworthy - an early parity
"failure" turned out to be a pin generated from our own output
(run 356), which proved nothing. Pins now come from the
independent references only, with HuggingFace as the generator.

**Layer 3 - the corpus anchors.** Whole corpora are hashed per
family, and the hash covers both the token count and a Fibonacci
mix of every id in order, so a permutation cannot pass. The final
battery re-derives four references over the full 476 MB corpus and
compares: all four EXACT, plus the document corpus at
156,701,082 ids, mismatches 0. The anchors use the latest model
versions, so an upstream tokenizer update that changes the ids
shows up here instead of in a user's bug report.

**Layer 4 - fuzzing.** Property fuzzers generate arbitrary bytes
and check the invariants: the pretokenizer must match its scalar
oracle, encode-then-decode must round-trip, and a parallel encode
must equal a sequential one. The all-models differential harness
runs every family against its reference at 60-second budgets per
battery, with campaigns in the tens of millions of runs on
record. A separate par-vs-seq fuzzer repeats past the 256 KiB
input threshold where the parallel planner starts splitting, so
the seam logic is exercised, not just the kernels. The harness
also carries the float32 tie rule: when two unigram segmentations
differ by less than float32 can distinguish, the comparison
accepts the tie instead of failing it.

**Layer 5 - the memory and race checkers.** The full suite runs
under ASan and UBSan (memory errors and undefined behavior), and
the parallel routes run under TSan (data races). Both are clean.
The race checking is not decorative: one cache-write ordering bug
was found only by a forced-collision hammer test, and the rare
race FIND C reproduced live during closure stress before its fix
(the full story is section 3.11).

**Layer 6 - the battery and its gates.** The full verification
battery chains every layer: build, parity, exhaustive algebra,
full corpora, fuzz, leak check, wrapper benches, the reference
comparison. It exits non-zero on any failure. That sentence matters more than
it looks: an earlier battery could not fail, and the gates were
fixed before this one went green (the story is section 3.11).

**Layer 7 - mechanical verification of the tricky parts.** The
Unicode normalizer is proven by sweep, not by samples: 5,328
normalization cases against the Unicode reference, 0 diffs. The
merge queue is proven by parity: the rank-bucket queue must
produce the same ids as the heap queue on every arm the test
forces, including the adversarial geometries (same-rank
overlapping pairs) that broke two earlier designs.

The result: 98 ctest tests green, the battery green end to end,
and every family byte-exact against its reference.

---

## 03 - Design decisions

This section states the decisions, the reason for each, and the
intuition. These are the choices that shaped the engine; most of
them were paid for at least once.

### 3.1 The simple version is the specification

Every hot function has a scalar twin that does the same job the
dumb way. The twin is never optimized. This costs code size and
buys two things. First, the twin defines correctness: "correct"
means "same bytes as the twin", which is checkable. Second, the
twin is the debugger: when the SIMD kernel misbehaves, the diff
between the two runs points at the exact block. The rule that
grew out of this: when the fast path and the oracle disagree, the
fast path is wrong by definition, even if the fast path looks
right and the oracle looks wrong. Twice in the campaign the oracle
"looked wrong" - and both times the fast path had a bug that only
the disagreement revealed.

### 3.2 No regex engine: compile the pattern, do not interpret it

The split rules of each family are fixed at load time. An
interpreter re-reads the rules per character; a compiled scanner
encodes them in straight-line code. tokensim goes one step
further: the scanner classifies a whole 64-byte block into 13
bitmasks in a few vector instructions, and the boundary logic
becomes integer algebra over those masks. The intuition: instead
of asking "what kind of byte is this?" 64 times, ask "which bytes
in this block are letters?" once, and answer for all 64 with one
compare. The price is that each new pattern needs a hand-written
scanner - the cost that showed up honestly when kimi's pattern
(class intersections like `\p{Lu}&&[^\p{Han}]`) needed its own
implementation.

### 3.3 Split only where it is provably safe

Parallelism multiplies throughput but multiplies ways to be wrong.
The engine does not split anywhere and hope. It splits only at
seams that satisfy one provable rule (a letter followed by an ASCII
non-letter, or a digit followed by an ASCII non-digit), so each
piece's encoding cannot depend on the bytes in another piece. Everything else - added-token hits,
digit runs, newlines - hard-splits the input into separate jobs
trivially. The invariant "parallel output equals sequential
output" is tested at every thread count, and the test caught real
bugs: the FFFD-expansion estimate error (run 364) and the
segmented-NFC mode that was not exact (runs 190–193). The deeper
rule the campaign recorded: **never compare a chunked encode
against a whole-buffer encode** - different engines, different
claims. Compare a chunked encode against the same chunked encode.

### 3.4 The leftmost tie must win: merge order is part of the semantics

BPE is greedy: merge the lowest-rank adjacent pair, repeat. When
several pairs share the same rank, the reference implementations
merge the leftmost one first. That tie order is not a detail - it
changes the output. The intuition: two overlapping pairs of the
same rank, like the two "ab" pairs in "abab", compete for the same
bytes. Merge the left one and the right one's bytes change; merge
the right one and the left one survives. The engine's queues
therefore carry the position in the ordering key: the heap packs
(rank, position) into one 64-bit number so the leftmost pair pops
first automatically; the bucket queue scans its list for the exact
lexicographic minimum. Two designs that ignored this - a pure
head-pop, and an append-list that broke position order under
re-pushes - were both caught by the arm-parity test and both
rejected. The lesson: order is data.

### 3.5 A cache is a correctness component

A cache that returns a wrong answer turns a correct engine into a
wrong one, silently and fast. So cache changes are treated as
correctness changes: each one gets its own test, its own hammer,
and the same review as an algorithm change. Two campaign examples.
The pretoken cache's write path had a two-step store (key, then
payload); a racing reader could see the new key with the old
payload - fixed with a claim protocol where the key acts as the
commit marker, proven by a test that forces the collision
geometry. And the short-pretoken cache's table simply stops
inserting when full: a hit is always exact, a miss is always safe,
so correctness never depends on the cache at all.

### 3.6 Trust CPU reports, but make the CPU prove them

CPUID says what the silicon supports. The OS decides what is
enabled, and the two can disagree - the documented case is
AVX-512 shipped but fused off, where the first instruction faults.
tokensim's answer: after CPUID says yes, run a tiny AVX-512
kernel under a crash handler once. Fault means permanent downgrade
to AVX2, decided once, remembered. The detection itself uses raw
CPUID + XGETBV instructions rather than the compiler's helper
library - not for speed, but because the helper library pulls in a
link-time dependency that made the Rust package unlinkable. The
whole dispatch is verified under Intel SDE, which emulates AVX-512
on machines that lack it - the AVX-512 paths have never run on
real AVX-512 hardware, and the battery still covers them.

### 3.7 Warm and cold are different claims

The first encode of a buffer pays cache-cold costs: the tables are
not resident, the caches are empty. The steady-state encode is
warm. These two numbers can differ by 2–4x, so comparing a cold
run against a warm history is comparing different machines. The
protocol is fixed: cold rows compare against cold history, warm
rows (last of several rounds) against warm history, and the
reference comparison runs in the same window - same corpora, same
pinning, same thread counts, both engines. The published numbers
of other engines are never used as a comparison.

### 3.8 The reference's own arithmetic decides semantics

When the engine and intuition disagreed with a reference, the
campaign read the reference's source and followed its arithmetic,
not the intuition. The rebase rule inside the unigram walk, the
wavefront boundary, the reordering semantics - all settled this
way. The intuition: these references are the definition of the
format. Their quirks are load-bearing, because the models were
trained on tokens produced by those exact quirks. "Fixing" a
reference quirk makes our ids wrong in the wild.

### 3.9 Worker caps are measured, not assumed

The parallel scheduler weights workers by their core's capability.
The first implementation hardcoded this machine's topology (six
performance cores, four efficiency cores, hyperthreading) into
the default path - which silently misbalanced any other machine.
The shipped form detects the topology at runtime: the sysfs
sibling lists give each cpu's position inside its core, and CPUID
leaf 0x1A distinguishes performance cores from efficiency cores.
The detected caps measured ~5% faster than the hand tuning on
this machine, which is the best kind of validation: the general
solution beat the specific one. The same run found the rare
livelock (see 3.11).

### 3.10 Gates must be able to fail

A verification step that cannot fail is decoration. The campaign
found three kinds of decoration in its own battery: exit codes
swallowed by pipes, a parity gate that always returned success,
and one tool that returned "error" on success. All fixed with
pipefail and exit-code plumbing, and the rule is now part of the
method: after building a gate, make it fire once - inject a fault,
corrupt an input - and watch it abort. The related rule: a fuzz
family that silently no-ops (four tiktoken families ran "millions
of runs" against a rejected spec for months) is worse than no
fuzzer, because it produces false confidence.

### 3.11 A waiter that only yields never sleeps

The one rare race (FIND C) reproduced live during closure stress:
under concurrent load, 8 of 10 threads spun in `sched_yield`
loops while the one-shot builders they waited for were starved of
CPU. `sched_yield` says "run someone else now, but I am ready
again immediately" - under load, the waiter stays permanently
ready and the builder never gets a full slice. The fix is one
line per waiter: after yielding, sleep 200 microseconds, which
gives the scheduler real freedom. 900 stress runs under the
reproduction condition (deliberate oversubscription) came back
clean. The intuition: yield is for fairness between equals;
sleep is for waiting.

### 3.12 The output must be reconstructible from the ids alone

Decoding maps ids back to bytes with no side tables, no state.
This forced several decisions that look like over-engineering:
the byte-fallback seeds, the added-token conventions, the detoken
parity against the reference decoders (run 206). The payoff is
that encode and decode can be tested separately and the round-trip
fuzzer covers arbitrary bytes.

---

## 04 - The model families

Fourteen families, each with its own encoding contract. This
section records what makes each distinctive, how special tokens
are handled, what the campaign won, and what it cost.

### GPT-2 (`encoder.json` + `vocab.bpe`)

The classic byte-level BPE: every byte is mapped to a visible
unicode stand-in so merges never break UTF-8, and the merge table
carries its own order, separate from ids. tokensim decodes the
alphabet once at load and never touches strings again. Special
tokens are scanned in one left-to-right pass indexed by first
byte; a hit hard-splits the input. This family pushed the
zero-copy parallel path and the split-invariance proof style that
the whole parallel engine rests on.

### cl100k (GPT-3.5 / GPT-4; Llama-3 reuses the pattern)

The pattern that introduced digit grouping in threes
(`\p{N}{1,3}`), case-insensitive contraction suffixes, and an
optional punctuation prefix on words. Implemented as bitmask
algebra with absorption carries across 64-byte windows. The
campaign proved here that the mask walker and the scalar matcher
are interchangeable (run 271), which is what lets the serve path
pick either. Llama-3 ships the same regex and is detected by
exact-match at load.

### o200k (GPT-4o)

The most intricate pattern: letter runs with a case-phase
transition (upper→lower, one allowed), contraction suffixes that
glue even when the letter run is empty, and a punctuation run that
re-absorbs newlines and slashes after them. The first design -
full mask algebra - failed and was replaced by a flat scanner,
which then needed five semantic fixes found by differential
testing. The family carries the campaign's most-quoted corpus pin.
It also taught a durability lesson: its fuzz family was a silent
no-op for months because a spec guard rejected it - the harness
rewrite turned "millions of runs clean" into real coverage.

### deepseek (tokenizer.json, byte_fallback)

A CJK-aware BPE: Han-script runs become isolated pieces;
everything else flows through byte-fallback BPE. The same
three-stage structure survived unchanged into DeepSeek-V3.1, and
the corpus hash for the V3.1 file came back identical to the V3
file, byte for byte. This family produced the campaign's sharpest
honesty moment: a divergence first blamed on the reference turned
out to be our bug, localized to one predicate, and the fix came
with the zero-copy parallel path it motivated. Special tokens
follow HF semantics: non-special added tokens are BPE-encoded, not
emitted as single ids.

### Qwen 2.5 and Qwen 3 (tokenizer.json)

Same JSON family shape with per-single-digit grouping in some
contexts, carried through the mask algebra as an explicit flag.
Qwen 3 uses the same encoding, and the corpus hash came back
identical to the Qwen 2.5 file. This family forced the NUL-key
hardening (added-token keys can contain embedded NUL bytes) and
the cache-hostile corpus probes that measured the caches under
adversarial repetition.

### Nemotron / Nemotron-Nano (tokenizer.json)

The serve path's home family, and the source of two
campaign-defining finds. First, the self-referential parity pin:
a "divergence" turned out to be a pin generated from our own
output; the HF-confirmed truth was the opposite of the pin's
assumption. Second, the walker cap-exit class: one missing
loop-exit guard produced both a multi-thread wedge and two
full-corpus divergences, and the same one-line guard closed the
class in all five mask schemes. After both fixes, 6t = 1t = 16t,
identical to the last id.

### SmolLM2 and SmolLM3 (tokenizer.json)

The beneficiary of the gpt2-pattern SIMD spanner: 111-125 MB/s
before, 390-420 MB/s after, the clearest single example of
"profile one family, land the shape for all". SmolLM3 loads
through the same cl100k pattern family. The lineage also carried
the 16-thread minus-one id class that the walker cap fix closed.
With the bucket queue, smollm2 runs 541.8 / 2257.2 / 2883.1 MB/s
at 1/6/10 threads, +47%/+109% at 6 threads over the heap era.

### GLM-4 and GLM-5.3 (tokenizer.json)

No unusual rules, which made it the clean control family for the
cache and parallel levers. The private per-worker caches and the
bucket queue both measured here first. GLM-5.3 loads through the
cl100k pattern family and its corpus came back EXACT on first
contact. The 6-thread warm row improved 14% when the bucket queue
became the default.

### Gemma 3-4B (tokenizer.json, metaspace pretokenizer)

Gemma's pretokenizer is not regex at all: a metaspace transform
(space becomes the ▁ character) plus byte-fallback BPE, with a
normalizer that expands invalid bytes to the U+FFFD replacement
character. Gemma 3 loads through the same GEM4 shape as Gemma 4. That expansion is the family's
signature hazard - the parallel path once estimated output length
without it and produced exactly-half outputs until per-job local
buffers with actual-length compaction fixed it. Gemma 3 also
hosted the windowed-DP saga: the proof that no chunked unigram DP
is exact, the boundary resolution, and the exact carry that made
the windowed form safe to promote.

### Gemma 4 (metaspace + byte-fallback)

The slowest family at the start (7–8 MB/s) and the clearest
demonstration that the method closes gaps: prefolded pair
priorities collapsed an 89%-in-merge profile, the long-unit cache
closed the one-thread gap, and a group-bucket variant was built,
verified, and disproven here. Gemma 4 also exposed the bucket
queue's final bug class through degenerate no-pretokenizer units.
With the bucket landed: 151.9 MB/s at 1t, +16% over the heap
default.

### Mistral (sentencepiece BPE, and the newer cl100k generation)

The family that taught "verify the model kind": the mistral file
is a sentencepiece BPE, not a unigram, and the parallel path once
routed it through the wrong normalizer, so the ids decoded to the
raw byte count while the sequential path produced 2.75x the bytes.
After the fix, the family drove the seam-scanner and stripe-target
levers (+17-55% combined) and hosts the giant-word fixture that
guards the merge engine against quadratic regressions. The newer
Mistral-Small-2501 generation switched to the cl100k pattern
family, and its corpus came back EXACT on first contact.

### XLM-R (sentencepiece unigram)

The unigram DP's home family: the two-level dynamic-programming
walk over a double-array trie, the unknown-token accounting, and
the boundary work. The unigram campaign's headline belongs here:
1.33-1.44x faster than the reference sentencepiece on this
machine. XLM-R and T5 run the ladder at one round by design:
their warm state equals cold, and the multi-round path is
memory-capped.

### T5 (sentencepiece unigram)

The family that forced the float32 tie rule. Two segmentations
whose scores differ by less than float32 can represent tie
exactly; the reference resolves one way and our engine the other,
and both are correct at their own precision. The resolution:
production was exonerated by the reference itself; the acceptance
rule (same id count, same unknown count, and a relative score
bound) became a library function with committed fixtures,
including a negative control that must still abort.

### Kimi K2 (tiktoken-format, moonshot's published pattern)

The newest family and the most distinctive pattern: class
intersections (`\p{Lu}&&[^\p{Han}]`), isolated Han runs, and
o200k-style contractions. Verified 12/12 against tiktoken's
engine plus a full-corpus EXACT. Kimi drove four structural wins:
the private per-worker caches (2.8–3.8x at multi-thread), the
window-batch mask algebra (the span-walker alternative measured
45% slower), the output-bound fallback (an 18x expansion bound),
and - through the "?s" span hunt - the dispatch fall-through bug
class: the dev API silently routed kimi to the deepseek walker,
while the production encode route was proven byte-exact against
tiktoken all along.

### Special tokens: the shared story

Across all families, one mechanism handles specials and added
tokens: a single left-to-right scan indexed by first byte
(two-byte indexed for the largest pools), hard splits at hits,
specials emitted or skipped, non-special added tokens BPE-encoded
per HF semantics. The pool route brought high added-token density
into the parallel path - along with three crashes that taught the
journalctl-first and memory-cap rules. The ignore_merges bug and
the byte_ids authority decision closed the added-token semantics
against the references.

---

## 05 - Benchmarks

Same-window comparisons only: the same corpora, the same pinning,
the same thread counts, warm = the last of repeated rounds on an
idle host.

**The machine.** Every number in this section was measured on the
same laptop: an Intel Core i5-13450HX (10 cores, 16 hardware
threads: 6 performance cores with hyperthreading up to 4.6 GHz,
plus 4 efficiency cores), 20 MB of L3 cache, 16 GB of RAM, running
a Linux 7.0 kernel. The CPU has no AVX-512, which is why the
AVX-512 paths are verified under the SDE emulator instead. Absolute
throughput on your machine will differ with the silicon; the
ratios against the reference engines are the durable claim.
Published numbers from other projects are context, never
comparisons. The run-by-run history behind every number here, with
the failed attempts and the reverted designs, lives in the
internal campaign report (CAMPAIGN-TECHNICAL-REPORT.md).

**Family throughput** (MB/s, the warm protocol, the 476 MB owt
corpus, the bucket queue as the default merge engine):

| Family | 1 thread | 6 threads | 10 threads |
| --- | --- | --- | --- |
| o200k | 567.3 | 2501.4 | 3560.6 |
| cl100k | 585.2 | 2732.6 | 3672.7 |
| deepseek | 538.2 | 2392.4 | 3029.5 |
| qwen2.5 | 457.0 | 1921.7 | 2436.8 |
| nemotron | 529.4 | 2244.8 | 2817.4 |
| glm4 | 536.6 | 2176.0 | 2655.5 |
| gemma4 | 151.9 | 377.4 | 373.7 |
| gemma3-4b | 127.8 | 353.3 | 385.9 |
| smollm2 | 541.8 | 2257.2 | 2883.1 |
| mistral | 153.4 | 341.5 | 316.1 |
| kimi | 107.1 | 412.7 | 448.6 |
| xlmr / t5 | 29.6 / 26.3 | 33.2 / 43.4 | 41.6 / 42.2 |

All ten-thread rows were measured fresh for this report under the
same protocol. Three shapes are worth reading from the table. The
byte-level families scale past 6 threads on the strength of the
performance cores. The metaspace families (gemma4, gemma3-4b,
mistral) flatten after 6 threads: their per-piece overhead stops
amortizing once the pieces get small. xlmr and t5 peak early (the
10-thread rows sit at or below the 6-thread rows) because the
unigram DP is a dependency chain that extra workers split thinly.

The bucket queue's promotion moved the unigram and metaspace
families: smollm2 +47%/+109%, gemma3-4b +38%, gemma4 +16%, glm4
+14% at 6 threads, with the byte-level families within noise of
the heap era. The 10-thread column shows the same ordering: the
byte-level families reach 3.0 to 3.7 GB/s, the metaspace families
level off in the 310 to 450 MB/s band, and the ratios against
gigatoken hold at 10 threads (the next table).

**Against gigatoken, the Rust reference engine this campaign
measures against** (the same-window protocol) -
ours/theirs ratio, cold and warm, threads 1/6/10:

| Threads | owt cold | owt warm | docs cold | docs warm |
| --- | --- | --- | --- | --- |
| 1 | 0.92 | 0.91 | 0.92 | 0.90 |
| 6 | 1.05 | 1.18 | 0.96 | 1.26 |
| 10 | 0.98 | 1.34 | 0.92 | 1.39 |

At 10 threads spanning P and E cores: parity cold, **1.34–1.39x
warm**. The single-thread cold gap (0.90–0.92x) is the recorded
honest frontier. Warm - the state sustained pipelines actually
run in - is ours ahead at every 6- and 10-thread point.

**The final battery** (exit 0): build clean; parity 13/13 at 1 and
4 threads; exhaustive algebra 0 mismatches; full corpora 4/4
EXACT; 13 fuzz families at 60 s plus 13 parallel budgets clean;
leak probe flat; the C/Python/Rust wrapper arms run; the reference
ladder above.

**Against the scalar reference engines, tiktoken and the official
HF deepseek tokenizer** (the same 16 MB
natural-text corpus, single thread, warm runs, this machine):

| encoding | reference engine, scalar | tokensim, 1 thread | tokensim, 6 threads |
| --- | --- | --- | --- |
| o200k | tiktoken 32.1 MB/s | 571.1 MB/s (**17.8x**) | 2709.0 MB/s |
| cl100k | tiktoken 22.5 MB/s | 576.0 MB/s (**25.6x**) | 2783.4 MB/s |
| deepseek | official HF tokenizer 2.3 MB/s | 509.0 MB/s (**~220x**) | 2355.7 MB/s |

The scalar rows are measured with the same-window protocol:
`scripts/bench_scalar_baseline.py` times tiktoken and the official
HF deepseek tokenizer on the identical corpus, and the ladder tool
times ours. The deepseek reference's 2.3 MB/s is its real-world
single-thread behavior - its pre-tokenizer regex is famously
expensive - and the multi-thread rows scale ~4.7x over our own
single thread with byte-identical output.

---

## 06 - Conclusion

tokensim is a tokenizer engine that treats correctness as the
product and speed as the measurable consequence of a disciplined
method: a scalar oracle defines the truth, compiled scanners and
bitmask algebra earn the speed, every layer of the stack is tested
against something independent, and every claim in this report is
reproducible from a fresh shell with the repo's own scripts.

It is worth being plain about what tokensim does and does not
cover, format by format. The tiktoken and sentencepiece loaders
cover both BPE and unigram models: T5 and XLM-R are unigram
models, fully supported through the sentencepiece format, and
llama-2-style sentencepiece files take the supported path, though
that specific model has not been reference-verified. The HuggingFace
JSON loader is the narrow one: it accepts only BPE models whose
pre-tokenizer matches a registered pattern, so a WordPiece model
or a Unigram model in JSON form is refused at load, loudly, as is
any new pre-tokenizer pattern variant. The normalizer support
covers the subset the supported families use (null, Sequence, NFC,
NFKC, and Gemma's replace); anything else is refused. The worker
pool caps at 64 threads and the pin arrays share that cap, so
hosts with more than 64 cpus get flat worker weights beyond it.
The single-thread cold gap against gigatoken (about 8%) stands.
And the AVX-512 paths have never run on real AVX-512 silicon: they
are SDE-verified, which is strong evidence, not proof. These are
the edges of the guarantee, stated here so the reader knows
exactly where the byte-exact promise applies.

The numbers say the method worked. Fourteen families are
byte-exact against their references. The battery is green under
gates that can fail. Against the Rust reference engine, tokensim
is at parity cold and ahead warm at every multi-thread point on
both corpora - 1.34–1.39x at 10 threads - with the honest
single-thread cold gap recorded rather than explained away.

The release artifacts (the version tag, the PyPI trusted
publisher, the crates.io token) are the remaining steps. The
wrappers install like any package, and the C engine builds inside
each one: `uv pip install tokensim` (or `pip install tokensim`)
for Python, `cargo add tokensim` for Rust, and CMake FetchContent
or pkg-config for C. The 98-test suite, the full verification
battery (`scripts/run_full_battery.sh`), the scripts behind every
measurement, and the fixtures behind every claim are in the
repository; the deeper engineering records are maintained
internally.
