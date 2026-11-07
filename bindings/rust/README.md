# tokensimd (Rust)

A safe Rust wrapper for TokenSIMD. All tokenization runs in the C library;
this crate only maps the stable `tokensimd_sys` C ABI.

```rust
let mut tok = tokensimd::Tokenizer::open(
    "models/cl100k_base.tiktoken",
    tokensimd::Format::Tiktoken,
    Some("cl100k"),
)?;
let ids = tok.encode(b"Hello world")?;
let text = tok.decode(&ids)?;
```

`cargo build` configures and builds the C library with CMake (Release).
Set `TOKENSIMD_LIB_DIR` to a directory holding a prebuilt
`libtokensimd.a` to skip that step, for example a `cmake --install`
prefix.

Run the tests with `cargo test`.

## Repository URL

The crate's `repository` field is `https://github.com/tokensimd/tokensimd`
— the release-time placeholder until the public repository lands.
