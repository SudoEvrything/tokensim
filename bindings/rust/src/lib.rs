use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;

#[repr(C)]
struct TsimdSys {
    _private: [u8; 0],
}

extern "C" {
    fn tsimd_sys_open(
        model_path: *const c_char,
        format: *const c_char,
        scheme: *const c_char,
        err: *mut c_char,
        err_cap: usize,
    ) -> *mut TsimdSys;
    fn tsimd_sys_close(h: *mut TsimdSys);
    fn tsimd_sys_vocab_size(h: *const TsimdSys) -> usize;
    fn tsimd_sys_version() -> *const c_char;
    fn tsimd_sys_encode(
        h: *mut TsimdSys,
        data: *const u8,
        len: usize,
        threads: c_int,
        ids: *mut *mut u32,
        n_ids: *mut usize,
    ) -> c_int;
    fn tsimd_sys_encode_into(
        h: *mut TsimdSys,
        data: *const u8,
        len: usize,
        threads: c_int,
        out: *mut u32,
        cap: usize,
        n_out: *mut usize,
    ) -> c_int;
    fn tsimd_sys_encode_own(
        h: *mut TsimdSys,
        data: *const u8,
        len: usize,
        threads: c_int,
    ) -> c_int;
    fn tsimd_sys_ids(h: *const TsimdSys, n: *mut usize) -> *const u32;
    fn tsimd_sys_encode_bound(
        h: *mut TsimdSys,
        data: *const u8,
        len: usize,
        threads: c_int,
        cap_out: *mut usize,
    ) -> c_int;
    fn tsimd_sys_decode(
        h: *mut TsimdSys,
        ids: *const u32,
        n: usize,
        skip_special: c_int,
        bytes: *mut *mut u8,
        n_bytes: *mut usize,
    ) -> c_int;
    fn tsimd_sys_free(p: *mut c_void);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {

    Json,

    Tiktoken,

    SentencePiece,

    Gpt2,
}

impl Format {
    fn as_str(self) -> &'static str {
        match self {
            Format::Json => "json",
            Format::Tiktoken => "tiktoken",
            Format::SentencePiece => "sentencepiece",
            Format::Gpt2 => "gpt2",
        }
    }
}

/// A loaded tokenizer. All methods that encode or decode take `&mut self`,
/// because the C library owns caches and a worker pool behind the handle.
/// `encode` and `encode_threads` return a slice borrowed from the
/// tokenizer's own id buffer, so a second call invalidates the previous

pub struct Tokenizer {
    handle: *mut TsimdSys,
}

unsafe impl Send for Tokenizer {}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size())
            .finish()
    }
}

impl Tokenizer {
    /// Load a model. `scheme` selects the pretokenizer scheme; `None` uses
    /// the library's automatic detection.
    pub fn open(
        model_path: &str,
        format: Format,
        scheme: Option<&str>,
    ) -> Result<Tokenizer, String> {
        let path = CString::new(model_path).map_err(|_| "model path has a NUL".to_string())?;
        let fmt = CString::new(format.as_str()).unwrap();
        let sch = match scheme {
            Some(s) => Some(CString::new(s).map_err(|_| "scheme has a NUL".to_string())?),
            None => None,
        };
        let mut err = [0i8; 256];
        let handle = unsafe {
            tsimd_sys_open(
                path.as_ptr(),
                fmt.as_ptr(),
                sch.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                err.as_mut_ptr(),
                err.len(),
            )
        };
        if handle.is_null() {
            let msg = unsafe { CStr::from_ptr(err.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            return Err(if msg.is_empty() {
                "cannot load the model".to_string()
            } else {
                msg
            });
        }
        Ok(Tokenizer { handle })
    }

    pub fn vocab_size(&self) -> usize {
        unsafe { tsimd_sys_vocab_size(self.handle) }
    }

    pub fn encode(&mut self, data: &[u8]) -> Result<&[u32], String> {
        self.encode_threads(data, 1)
    }

    pub fn encode_threads(&mut self, data: &[u8], threads: usize) -> Result<&[u32], String> {
        self.encode_borrowed(data, threads)
    }

    fn encode_borrowed(&mut self, data: &[u8], threads: usize) -> Result<&[u32], String> {
        let rc = unsafe {
            tsimd_sys_encode_own(
                self.handle,
                data.as_ptr(),
                data.len(),
                threads.max(1) as c_int,
            )
        };
        if rc != 0 {
            return Err("encode failed".to_string());
        }
        let mut n: usize = 0;
        let p = unsafe { tsimd_sys_ids(self.handle, &mut n) };
        if p.is_null() || n == 0 {
            return Ok(&[]);
        }
        Ok(unsafe { std::slice::from_raw_parts(p, n) })
    }

    pub fn encode_into(
        &mut self,
        data: &[u8],
        out: &mut [u32],
        threads: usize,
    ) -> Result<usize, String> {
        let mut n: usize = 0;
        let rc = unsafe {
            tsimd_sys_encode_into(
                self.handle,
                data.as_ptr(),
                data.len(),
                threads.max(1) as c_int,
                out.as_mut_ptr(),
                out.len(),
                &mut n,
            )
        };
        if rc != 0 {
            return Err("encode failed (buffer too small?)".to_string());
        }
        Ok(n)
    }

    pub fn encode_bound(&self, len: usize) -> usize {
        let mut cap: usize = 0;
        unsafe { tsimd_sys_encode_bound(self.handle, ptr::null(), len, 1, &mut cap) };
        cap
    }

    pub fn decode(&mut self, ids: &[u32]) -> Result<Vec<u8>, String> {
        self.decode_opt(ids, false)
    }

    pub fn decode_opt(&mut self, ids: &[u32], skip_special: bool) -> Result<Vec<u8>, String> {
        let mut bytes: *mut u8 = ptr::null_mut();
        let mut n: usize = 0;
        let rc = unsafe {
            tsimd_sys_decode(
                self.handle,
                ids.as_ptr(),
                ids.len(),
                skip_special as c_int,
                &mut bytes,
                &mut n,
            )
        };
        if rc != 0 {
            return Err("decode failed (unknown id)".to_string());
        }
        let out = if n == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(bytes, n).to_vec() }
        };
        if !bytes.is_null() {
            unsafe { tsimd_sys_free(bytes as *mut c_void) };
        }
        Ok(out)
    }
}

impl Drop for Tokenizer {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { tsimd_sys_close(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

pub fn version() -> &'static str {
    let p = unsafe { tsimd_sys_version() };
    if p.is_null() {
        return "";
    }
    unsafe { CStr::from_ptr(p) }.to_str().unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str) -> String {
        let root = std::env::var("TOKENSIMD_MODELS_DIR").unwrap_or_else(|_| {
            format!("{}/../../models/reference", env!("CARGO_MANIFEST_DIR"))
        });
        format!("{}/{}", root, name)
    }

    fn models_available() -> bool {
        std::path::Path::new(&model("t5_spiece.model")).exists()
    }

    #[test]
    fn version_is_reported() {
        assert_eq!(version(), "1.0.0");
    }

    #[test]
    fn encode_into_matches_encode() {
        if !models_available() {
            eprintln!("skip: no models dir (set TOKENSIMD_MODELS_DIR)");
            return;
        }
        let mut tok = Tokenizer::open(&model("t5_spiece.model"), Format::SentencePiece, None)
            .expect("load t5");
        let data = b"Encode into a caller buffer, once. ".repeat(20000);
        let ids: Vec<u32> = tok.encode(&data).unwrap().to_vec();
        let mut out = vec![0u32; tok.encode_bound(data.len())];
        let n = tok.encode_into(&data, &mut out, 1).unwrap();
        assert_eq!(n, ids.len());
        assert_eq!(&out[..n], &ids[..]);
        let n4 = tok.encode_into(&data, &mut out, 4).unwrap();
        assert_eq!(n4, ids.len());
        assert_eq!(&out[..n4], &ids[..]);
        let mut small = vec![0u32; 8];
        assert!(tok.encode_into(&data, &mut small, 1).is_err());
    }

    #[test]
    fn round_trip_sentencepiece() {
        if !models_available() { eprintln!("skip: no models dir"); return; }
        let mut tok = Tokenizer::open(
            &model("t5_spiece.model"),
            Format::SentencePiece,
            None,
        )
        .expect("load t5");
        assert_eq!(tok.vocab_size(), 32000);
        let ids: Vec<u32> = tok.encode(b"Hello world, this is a test.").unwrap().to_vec();
        assert!(!ids.is_empty());
        let text = tok.decode(&ids).unwrap();
        assert_eq!(text, b"Hello world, this is a test.");
    }

    #[test]
    fn parallel_matches_sequential() {
        if !models_available() { eprintln!("skip: no models dir"); return; }
        let mut tok = Tokenizer::open(&model("t5_spiece.model"), Format::SentencePiece, None)
            .expect("load t5");
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(4000);
        let a: Vec<u32> = tok.encode(&data).unwrap().to_vec();
        let b: Vec<u32> = tok.encode_threads(&data, 4).unwrap().to_vec();
        assert_eq!(a, b);
        let back = tok.decode(&a).unwrap();
        assert!(back.starts_with(b"The quick brown fox jumps over the lazy dog."));
        assert!(back.len() >= data.len() - 1);
    }

    #[test]
    fn hf_pinned_ids() {
        if !models_available() { eprintln!("skip: no models dir"); return; }
        let mut tok = Tokenizer::open(
            &model("deepseek_tokenizer.json"),
            Format::Json,
            None,
        )
        .expect("load deepseek");
        let ids: Vec<u32> = tok.encode(b"Hello, world!").unwrap().to_vec();
        assert_eq!(ids, vec![19923, 14, 2058, 3]);
        let mut tok4 = Tokenizer::open(&model("gemma4_tokenizer.json"), Format::Json, None)
            .expect("load gemma4");
        let ids4: Vec<u32> = tok4.encode(b"Hello, world!").unwrap().to_vec();
        let back = tok4.decode(&ids4).unwrap();
        assert_eq!(back, b"Hello, world!");
    }

    #[test]
    fn missing_model_reports_an_error() {
        let err = Tokenizer::open("/nonexistent/model.json", Format::Json, None).unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn bad_scheme_reports_an_error() {
        if !models_available() { eprintln!("skip: no models dir"); return; }
        let err = Tokenizer::open(
            &model("t5_spiece.model"),
            Format::SentencePiece,
            Some("nonsense"),
        )
        .unwrap_err();
        assert!(err.contains("scheme"), "unexpected message: {err}");
    }
}
