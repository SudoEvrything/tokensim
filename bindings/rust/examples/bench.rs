use std::env;
use std::fs;
use std::time::Instant;

use tokensimd::{Format, Tokenizer};

fn main() {
    let a: Vec<String> = env::args().collect();
    if a.len() < 6 {
        eprintln!("usage: bench MODEL FORMAT SCHEME THREADS FILE [REPS]");
        std::process::exit(2);
    }
    let fmt = match a[2].as_str() {
        "tiktoken" => Format::Tiktoken,
        "spm" => Format::SentencePiece,
        "gpt2" => Format::Gpt2,
        _ => Format::Json,
    };
    let scheme = if a[3] == "auto" { None } else { Some(a[3].as_str()) };
    let mut tok = Tokenizer::open(&a[1], fmt, scheme).unwrap_or_else(|e| panic!("load {}: {}", a[1], e));
    let threads: usize = a[4].parse().unwrap();
    let data = fs::read(&a[5]).expect("read corpus");
    let reps: usize = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(1);
    let into = std::env::var("BENCH_INTO").is_ok();
    let mut buf = if into {
        vec![0u32; tok.encode_bound(data.len())]
    } else {
        Vec::new()
    };
    let mut best = 0.0f64;
    let mut n = 0usize;
    for r in 0..reps {
        let t0 = Instant::now();
        if into {
            n = tok
                .encode_into(&data, &mut buf, threads)
                .expect("encode_into");
        } else {
            n = tok.encode_threads(&data, threads).expect("encode").len();
        }
        let el = t0.elapsed().as_secs_f64();
        let mibs = (data.len() as f64 / 1048576.0) / el;
        if mibs > best {
            best = mibs;
        }
        println!("REP rust{} {} {:.1} {}", if into { "-into" } else { "" }, r,
                 mibs, n);
    }
    println!("RESULT rust {} {} {} {:.1} {}", a[3], a[5], threads, best, n);
}
