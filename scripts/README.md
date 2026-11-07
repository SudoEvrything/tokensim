# TokenSIMD test and measurement scripts

Everything here survives reboots; /tmp scratch binaries do not. Each
`run_*.sh` rebuilds what it needs from the repo and the current
`build/libtokensimd.a`, so `cmake --build build` first is the only
prerequisite.

| script | what it does |
|---|---|
| `run_ladder.sh` | throughput ladder, ours vs gigatoken, per family, 1t/6t; prints MB/s. Absolute rows are valid only on an idle host. |
| `run_fuzz.sh [sec]` | adversarial libFuzzer budgets for all nine families against the all-models oracle harness. `MISMATCH` lines mean a real encode/reference divergence; the failing input is saved as `crash-*` in the repo root. |
| `gate_parity.sh [4t]` | the nine-family id-hash parity gate at 1t (or 4t). Every line must print OK. The wanted hashes were derived from the reference encoders (tiktoken / HF tokenizers / the spm models) and must not change without a reference re-derivation. |
| `run_exhaustive.sh` | exhaustive law-vs-masks checker for the six mask schemes over all short strings plus boundary and random tiers. Any nonzero mismatch count is a bug. |
| `run_microbench.sh` | window-algebra cost in ns per 64-byte window for o200k and deepseek. Cache-resident, so it stays valid under host load. |
| `repro_artifact.sh` | replay one saved artifact against its family harness with debug output. |
| `ladder_encode.c` | the ladder driver source (built by run_ladder.sh). |
| `exhaustive_algebra.c` | the checker source. |
| `diag_compare.c` | four-path comparator on one input: scalar law, masks spans, the keyed fill, the encode versus the law BPE. Build: `gcc -O1 -I include scripts/diag_compare.c build/libtokensimd.a -lm -lpthread` and run `MODEL INPUT`. |
| `microbench_windows.c` | the window microbench source. |

The debug workflow for a pretokenizer divergence: `repro_artifact.sh`
for the id-level view, `diag_compare.c` for the span-level view, then
the term-level mask dumps (see docs/286 for the instrumented-probe
pattern).

## The warm protocol (2026-10-31)

run_ladder.sh sets ENCODE_ROUNDS=3 and reports the LAST round (the
warm pass). The pre-294 rows were the cold first pass and do not
compare against the warm history (the run-283 table). The cold rows
sit 30-40 percent below the warm rows on every family; the cache and
the page cache need one pass to warm.
