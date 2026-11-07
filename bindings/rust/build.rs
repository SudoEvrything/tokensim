use std::env;
use std::path::PathBuf;

// The per-file ISA flag table mirrors CMakeLists.txt:79-120. cc applies
// flags to the whole invocation, so each ISA group compiles separately
// and the plain group carries everything unflagged.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    let target = env::var("TARGET").unwrap_or_default();
    let x86 = target.contains("x86_64") || target.contains("i686");
    if !x86 {
        panic!(
            "TokenSIMD 1.x ships x86-64 kernels only (SSE2/AVX2/AVX-512); \
             this target ({target}) has no port yet."
        );
    }
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    if let Ok(dir) = env::var("TOKENSIMD_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
            link_libs(&target);
        return;
    }

    let mut objs: Vec<PathBuf> = Vec::new();
    let mut plain = cc::Build::new();
    plain
        .include("csrc/include")
        .include("csrc/src/simd")
        .include("csrc/src/pretokenizer")
        .flag_if_supported("-std=gnu17")
        .pic(true);
    for f in walk("csrc/src") {
        if is_flagged(&f) {
            continue;
        }
        plain.file(f);
    }
    objs.extend(plain.compile_intermediates());

    objs.extend(compile_group(
        &["-std=gnu17", "-fno-rtti", "-mavx2"],
        &[
            "csrc/src/pretokenizer/o200k_masks.c",
            "csrc/src/pretokenizer/cl100k_masks.c",
            "csrc/src/pretokenizer/nemo_masks.c",
            "csrc/src/pretokenizer/deepseek_masks.c",
            "csrc/src/pretokenizer/gpt2_masks.c",
            "csrc/src/pretokenizer/kimi_masks.c",
            "csrc/src/serve_x86.cpp",
            "csrc/src/simd/blocks_avx2.c",
        ],
    ));
    objs.extend(compile_group(
        &["-std=gnu17", "-msse2"],
        &["csrc/src/simd/blocks_sse2.c"],
    ));
    objs.extend(compile_group(
        &["-std=gnu17", "-mavx512f", "-mavx512bw", "-mavx512vl"],
        &["csrc/src/simd/blocks_avx512.c"],
    ));
    objs.extend(compile_group(&["-std=gnu17", "-msse4.2"], &["csrc/src/bpe.c"]));
    objs.extend(compile_group(&["-std=gnu17", "-mpopcnt"], &["csrc/src/nfc.c"]));

    println!("cargo:rustc-link-search=native={}", out.display());
    link_libs(&target);
    for o in &objs {
        println!("cargo:rustc-link-arg={}", o.display());
    }
}

fn link_libs(target: &str) {
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=m");

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

fn compile_group(flags: &[&str], files: &[&str]) -> Vec<PathBuf> {
    let mut b = cc::Build::new();
    b.include("csrc/include")
        .include("csrc/src/simd")
        .include("csrc/src/pretokenizer")
        .pic(true);
    for f in flags {
        b.flag_if_supported(f);
    }
    for f in files {
        assert!(PathBuf::from(f).exists(), "missing vendored source: {f}");
        b.file(*f);
    }
    b.compile_intermediates()
}

fn is_flagged(f: &PathBuf) -> bool {
    let p = f.to_string_lossy().replace('\\', "/");
    const FLAGGED: &[&str] = &[
        "csrc/src/pretokenizer/o200k_masks.c",
        "csrc/src/pretokenizer/cl100k_masks.c",
        "csrc/src/pretokenizer/nemo_masks.c",
        "csrc/src/pretokenizer/deepseek_masks.c",
        "csrc/src/pretokenizer/gpt2_masks.c",
        "csrc/src/pretokenizer/kimi_masks.c",
        "csrc/src/serve_x86.cpp",
        "csrc/src/simd/blocks_avx2.c",
        "csrc/src/simd/blocks_sse2.c",
        "csrc/src/simd/blocks_avx512.c",
        "csrc/src/bpe.c",
        "csrc/src/nfc.c",
    ];
    FLAGGED.iter().any(|fl| p.ends_with(*fl))
}

fn walk(dir: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![PathBuf::from(dir)];
    while let Some(d) = stack.pop() {
        let rd = match std::fs::read_dir(&d) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                if ext == "c" || ext == "cpp" {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out
}
