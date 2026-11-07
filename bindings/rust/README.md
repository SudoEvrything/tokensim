# tokensim

Rust bindings for the tokensim tokenizer engine: 14 model families,
byte-exact against tiktoken, HuggingFace tokenizers, and
sentencepiece. The C engine is vendored inside the crate and built
with cc, so `cargo add tokensim` works anywhere with no cmake, no
network access, and no system dependencies.

## Install

```sh
cargo add tokensim
```

## Encode and decode

```rust
use tokensim::{Format, Tokenizer};

fn main() -> Result<(), String> {
    let mut tok = Tokenizer::open(
        "cl100k_base.tiktoken",   // or any tiktoken / GPT-2 / tokenizer.json / sentencepiece model
        Format::Tiktoken,
        Some("cl100k"),           // or None to auto-detect where the format allows
    )?;

    let ids = tok.encode(b"hello world, don't stop")?;
    println!("{} ids", ids.len());

    let ids = tok.encode_threads(b"...", 8)?;      // parallel, byte-identical to encode
    let text = tok.decode(&ids)?;                  // round-trips exactly
    let text = tok.decode_opt(&ids, true)?;        // specials omitted
    let bound = tok.encode_bound(1024);            // the out-buffer bound
    println!("tokensim {}", tokensim::version());
    Ok(())
}
```

Three things to know:

- `encode` and `encode_threads` are byte-identical at any thread
  count; the thread argument is a performance hint, not a
  correctness knob.
- `decode_opt(.., true)` drops special tokens from the output.
- `encode_bound(len)` returns the required out-buffer size for the
  zero-copy `encode_into` form.

## Performance

Same-window protocol, 16 MB of natural text, this machine:

| encoding | reference engine, scalar | tokensim, 1 thread | tokensim, 6 threads |
| --- | --- | --- | --- |
| o200k | tiktoken 32.1 MB/s | 571.1 MB/s (**17.8x**) | 2709.0 MB/s |
| cl100k | tiktoken 22.5 MB/s | 576.0 MB/s (**25.6x**) | 2783.4 MB/s |
| deepseek | official HF tokenizer 2.3 MB/s | 509.0 MB/s (**~220x**) | 2355.7 MB/s |

Against gigatoken, the Rust reference engine: parity cold, ahead
warm at every 6- and 10-thread point.

## The model families

Fourteen families, each byte-exact against its reference:

| family | format | notes |
| --- | --- | --- |
| GPT-2 | encoder.json + vocab.bpe | the classic OpenAI byte-level BPE |
| cl100k | tiktoken | GPT-3.5/4; Llama-3 reuses the same regex |
| o200k | tiktoken | GPT-4o; Llama-4 and MiniMax-M2 share the pattern |
| deepseek | tokenizer.json | CJK-aware BPE with byte fallback |
| qwen2.5 / qwen3 | tokenizer.json | per-single-digit grouping |
| nemotron / nano | tokenizer.json | the serve path's home family |
| smollm2 | tokenizer.json | the gpt2-pattern SIMD spanner |
| glm4 | tokenizer.json | the clean control family |
| gemma3-4b / gemma4 | tokenizer.json | metaspace + byte-fallback BPE |
| mistral | sentencepiece BPE | the spm_kind=0 family |
| xlmr / t5 | sentencepiece unigram | the unigram DP |
| kimi-k2 | tiktoken | moonshot's pattern with class intersections |

## The API at a glance

| method | what it does |
| --- | --- |
| `Tokenizer::open(path, format, scheme)` | load a model file |
| `encode(data) -> &[u32]` | sequential encode |
| `encode_threads(data, n) -> &[u32]` | parallel encode, byte-identical |
| `encode_into(data, buf, cap) -> usize` | zero-copy into the caller's buffer |
| `encode_bound(len) -> usize` | the required out-buffer size |
| `decode(&[u32]) -> Vec<u8>` | ids back to bytes, exact round-trip |
| `decode_opt(&[u32], skip_special) -> Vec<u8>` | decode with the specials filtered |
| `vocab_size() -> usize` | the vocabulary size |
| `version() -> &'static str` | the engine version |

## Links

- [Repository](https://github.com/SudoEvrything/tokensim)
- [Technical report](https://github.com/SudoEvrything/tokensim/blob/main/TECHNICAL-REPORT.md) — the deep-dive engineering blog
- [README](https://github.com/SudoEvrything/tokensim#readme) — the full overview
