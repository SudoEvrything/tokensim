use std::time::Instant;
use tokensim::{Format, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = "/home/raising-reuel/tokenizer/models/cl100k_base.tiktoken";
    let corpus = "/home/raising-reuel/tokenizer/testdata/owt.txt";
    let mut tok = Tokenizer::open(model, Format::Tiktoken, Some("cl100k"))?;
    let data = std::fs::read(corpus)?;
    println!("input: {} MB", data.len() / (1024 * 1024));
    let mb = data.len() as f64 / 1e6;

    let t0 = Instant::now();
    let ids: Vec<u32> = tok.encode(&data)?.to_vec();
    let t1 = Instant::now();
    println!(
        "1t encode (to_vec): {:.1} MB/s ({} ids)",
        mb / (t1 - t0).as_secs_f64(),
        ids.len()
    );
    let rt = tok.decode(&ids)?;
    println!("round-trip: {}", rt == data);

    let bound = tok.encode_bound(data.len());
    let mut buf = vec![0u32; bound];

    for _ in 0..2 {
        tok.encode_into(&data, &mut buf, 1)?;
    }
    let t0 = Instant::now();
    let n1 = tok.encode_into(&data, &mut buf, 1)?;
    let t1 = Instant::now();
    println!(
        "1t into (zero-copy): {:.1} MB/s ({} ids)",
        mb / (t1 - t0).as_secs_f64(),
        n1
    );

    for _ in 0..2 {
        tok.encode_into(&data, &mut buf, 6)?;
    }
    let t0 = Instant::now();
    let n6 = tok.encode_into(&data, &mut buf, 6)?;
    let t1 = Instant::now();
    println!(
        "6t into: {:.1} MB/s ({} ids)",
        mb / (t1 - t0).as_secs_f64(),
        n6
    );

    for _ in 0..2 {
        tok.encode_into(&data, &mut buf, 16)?;
    }
    let t0 = Instant::now();
    let n16 = tok.encode_into(&data, &mut buf, 16)?;
    let t1 = Instant::now();
    println!(
        "16t into: {:.1} MB/s ({} ids)",
        mb / (t1 - t0).as_secs_f64(),
        n16
    );

    println!("parity: 1t==6t {} 1t==16t {}", n1 == n6, n1 == n16);
    Ok(())
}
