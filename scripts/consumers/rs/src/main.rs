use std::fs::File;
use std::io::Read;
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

fn rss_kb() -> i64 {
    let mut s = String::new();
    if File::open("/proc/self/statm")
        .and_then(|mut f| f.read_to_string(&mut s))
        .is_err()
    {
        return -1;
    }
    let mut it = s.split_whitespace();
    it.next();
    match it.next().and_then(|v| v.parse::<i64>().ok()) {
        Some(pages) => pages * 4,
        None => -1,
    }
}

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

    let mut h = FNV0;
    let mut total: u64 = 0;
    let mut checked: u64 = 0;
    let mut bad: u64 = 0;
    let mut docs: u64 = 0;
    let mut rss = [rss_kb(), -1i64, -1, -1, -1];
    let mut pos: u64 = 0;
    let mut owned: Vec<u32> = Vec::new();
    let mut do_check = false;
    let t0 = Instant::now();
    let mut off: usize = 0;
    loop {
        let n = usize::min(fsz as usize - off, doc_bytes);
        if n == 0 {
            break;
        }
        {
            let doc = &mmap[off..off + n];
            let ids = if threads != 0 {
                tok.encode_threads(doc, threads)?
            } else {
                tok.encode(doc)?
            };
            fold_count(&mut h, ids.len() as u64);
            let head = ids.len().min(4096);
            fold_ids(&mut h, &ids[..head]);
            if ids.len() > head {
                fold_ids(&mut h, &ids[ids.len() - head..]);
            }
            total += ids.len() as u64;
            owned.clear();
            do_check = (docs + 1) % 64 == 0 || n < doc_bytes;
            if do_check {
                owned.extend_from_slice(ids);
            }
        }
        docs += 1;
        if do_check {
            if tok.decode(&owned)? != mmap[off..off + n] {
                bad += 1;
            }
            checked += 1;
        }
        off += n;
        pos += n as u64;
        for q in 1..5u64 {
            if rss[q as usize] < 0 && pos * 4 >= fsz * q {
                rss[q as usize] = rss_kb();
            }
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    for i in 0..5 {
        if rss[i] < 0 {
            rss[i] = rss_kb();
        }
    }
    println!("input: {} bytes in {} docs of {} bytes", fsz, docs, doc_bytes);
    println!("ids: {}", total);
    println!("hash: 0x{:016x}", h);
    println!(
        "roundtrip: {} ({} checked)",
        if bad == 0 { "ok" } else { "MISMATCH" },
        checked
    );
    println!("docs: {:.3} s = {:.1} MB/s", elapsed, fsz as f64 / 1e6 / elapsed);
    println!(
        "rss: {} {} {} {} {} KB",
        rss[0], rss[1], rss[2], rss[3], rss[4]
    );

    let mut ph = FNV0;
    let mut pids: u64 = 0;
    let p0 = Instant::now();
    let batch_n = 256;
    let mut done: usize = 0;
    while done < prompt_calls {
        let n = usize::min(prompt_calls - done, batch_n);
        let chunk: Vec<&[u8]> = (0..n)
            .map(|i| PROMPTS[(done + i) % 8].as_bytes())
            .collect();
        let batch = tok.encode_batch(&chunk, threads)?;
        for ids in &batch {
            fold_count(&mut ph, ids.len() as u64);
            fold_ids(&mut ph, ids);
            pids += ids.len() as u64;
        }
        done += n;
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
