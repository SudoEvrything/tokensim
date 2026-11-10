use std::fs::File;
use std::time::Instant;
use memmap2::Mmap;
use tokensim::{default_threads, Format, Tokenizer};

const PRIME: u64 = 0x100000001B3;
const FNV0: u64 = 0xCBF29CE484222325;

const PROMPTS: [&str; 8] = [
    "Summarize the following article in one sentence.",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "The quick brown fox jumps over the lazy dog.",
    "Bonjour, comment allez-vous aujourd'hui ?",
    "Explain the difference between a list and a tuple in Python.",
    "SELECT name, COUNT(*) FROM users GROUP BY name ORDER BY 2 DESC;",
    "Machine learning models are trained on large corpora of text.",
    "The recipe calls for 2 cups of flour, 1 tsp of salt, and 3 eggs.",
];

fn fold_bytes(h: &mut u64, data: &[u8]) {
    for &b in data {
        *h ^= b as u64;
        *h = h.wrapping_mul(PRIME);
    }
}

fn fold_count(h: &mut u64, v: u64) {
    fold_bytes(h, &v.to_le_bytes());
}

fn fold_ids(h: &mut u64, ids: &[u32]) {
    for &id in ids {
        fold_bytes(h, &id.to_le_bytes());
    }
}

fn format_of(s: &str) -> Option<Format> {
    match s {
        "json" => Some(Format::Json),
        "tiktoken" => Some(Format::Tiktoken),
        "sentencepiece" => Some(Format::SentencePiece),
        "gpt2" => Some(Format::Gpt2),
        _ => None,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: {} MODEL FORMAT SCHEME CORPUS [DOC_BYTES] [PROMPT_CALLS] [THREADS]",
            args[0]
        );
        std::process::exit(2);
    }
    let doc_bytes: usize = if args.len() > 5 {
        args[5].parse().unwrap_or(16 << 20)
    } else {
        16 << 20
    };
    let prompt_calls: usize = if args.len() > 6 {
        args[6].parse().unwrap_or(20000)
    } else {
        20000
    };
    let threads: usize = if args.len() > 7 {
        args[7].parse().unwrap_or(0)
    } else {
        0
    };
    let fmt = format_of(&args[2]).ok_or("unknown format")?;
    let mut tok = Tokenizer::open(&args[1], fmt, Some(&args[3]))?;
    println!("threads(auto): {}", default_threads());

    let fsz = std::fs::metadata(&args[4])?.len();
    let mmap = unsafe { Mmap::map(&File::open(&args[4])?)? };
    mmap.advise(memmap2::Advice::Sequential)?;

    let mut h = FNV0;
    let mut total: u64 = 0;
    let mut checked: u64 = 0;
    let mut bad: u64 = 0;
    let mut docs: u64 = 0;
    let mut enc_t = 0f64;
    let mut dec_t = 0f64;
    let t0 = Instant::now();
    let mut off: usize = 0;
    loop {
        let n = usize::min(fsz as usize - off, doc_bytes);
        if n == 0 {
            break;
        }
        let doc = &mmap[off..off + n];
        let a = Instant::now();
        let ids = tok.encode(doc)?.to_vec();
        enc_t += a.elapsed().as_secs_f64();
        fold_count(&mut h, ids.len() as u64);
        let head = ids.len().min(4096);
        fold_ids(&mut h, &ids[..head]);
        if ids.len() > head {
            fold_ids(&mut h, &ids[ids.len() - head..]);
        }
        total += ids.len() as u64;
        docs += 1;
        let c = Instant::now();
        let back = tok.decode(&ids)?;
        dec_t += c.elapsed().as_secs_f64();
        checked += 1;
        if back.len() != n || &mmap[off..off + n] != back.as_slice() {
            bad += 1;
        }
        off += n;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    println!("input: {} bytes in {} docs of {} bytes", fsz, docs, doc_bytes);
    println!("ids: {}", total);
    println!("hash: 0x{:016x}", h);
    println!(
        "roundtrip: {} ({} checked)",
        if bad == 0 { "ok" } else { "MISMATCH" },
        checked
    );
    println!("encode: {:.3} s = {:.1} MB/s", enc_t, fsz as f64 / 1e6 / enc_t);
    println!("decode: {:.3} s = {:.1} MB/s", dec_t, fsz as f64 / 1e6 / dec_t);
    println!(
        "docs (round trip): {:.3} s = {:.1} MB/s",
        elapsed,
        fsz as f64 / 1e6 / elapsed
    );

    let mut ph = FNV0;
    let mut pids: u64 = 0;
    let p0 = Instant::now();
    let chunk: Vec<&[u8]> = (0..prompt_calls)
        .map(|i| PROMPTS[i % 8].as_bytes())
        .collect();
    let batch = tok.encode_batch(&chunk, threads)?;
    for ids in &batch {
        fold_count(&mut ph, ids.len() as u64);
        fold_ids(&mut ph, ids);
        pids += ids.len() as u64;
    }
    let pel = p0.elapsed().as_secs_f64();
    println!(
        "prompts: {} calls in {:.3} s = {:.0} calls/s, {} ids, hash 0x{:016x}",
        prompt_calls,
        pel,
        prompt_calls as f64 / pel,
        pids,
        ph
    );
    Ok(())
}
