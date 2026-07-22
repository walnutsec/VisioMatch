//! ArcFace (MobileFaceNet) inference, embedding math, and JSON persistence.
//!
//! This module wraps the `w600k_mbf.onnx` ArcFace model behind a thin
//! [`ArcFaceNet`] struct that caches the DNN output-layer names so they are
//! resolved once at load time rather than on every forward pass.

use opencv::{
    core::{self, Mat, Rect, Scalar, Size, Vector},
    dnn, imgproc,
    prelude::*,
};
use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::Path,
};

use crate::Result;

// ── Constants

/// Filesystem path to the ArcFace ONNX model (relative to project root).
pub const ARCFACE_MODEL: &str = "data/models/w600k_mbf.onnx";

/// The ArcFace model expects a 112×112 RGB face crop.
const ARCFACE_INPUT_SIZE: i32 = 112;

/// Dimensionality of the ArcFace embedding vector.
pub const EMBEDDING_DIM: usize = 512;

/// Cosine-similarity threshold for declaring a match.
///
/// ArcFace same-person pairs typically land in **0.4–0.8**; cross-person
/// pairs are usually **below 0.3**.  Raise for stricter matching, lower to
/// reduce false rejections.
pub const COSINE_THRESHOLD: f64 = 0.4;

/// Filesystem path to the JSON embedding database (relative to project root).
pub const EMBEDDINGS_PATH: &str = "data/embeddings.json";

// ── ArcFaceNet wrapper ─────────────────────────────────────────────────

/// Wraps an OpenCV `dnn::Net` loaded with the ArcFace ONNX model and caches
/// the unconnected-output-layer names so they are resolved once at load time
/// rather than on every [`ArcFaceNet::forward`] call.
pub struct ArcFaceNet {
    net: dnn::Net,
    out_names: Vector<String>,
}

impl ArcFaceNet {
    /// Load the ArcFace ONNX model from [`ARCFACE_MODEL`] and configure the
    /// OpenCV DNN backend.
    ///
    /// Uses `DNN_BACKEND_OPENCV` + `DNN_TARGET_OPENCL` to offload inference to
    /// the GPU via OpenCL when available (falls back to CPU transparently).
    pub fn load() -> Result<Self> {
        if !Path::new(ARCFACE_MODEL).exists() {
            return Err(format!(
                "ArcFace model not found at: {ARCFACE_MODEL}\n\
                 Download it:\n\
                   wget -O data/models/w600k_mbf.onnx \\\n\
                     https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx"
            ).into());
        }
        let mut net = dnn::read_net_from_onnx_def(ARCFACE_MODEL)?;
        net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
        net.set_preferable_target(dnn::DNN_TARGET_OPENCL)?;

        // Cache once — this queries the DNN graph topology which is static.
        let out_names = net.get_unconnected_out_layers_names()?;

        println!("[*] ArcFace model loaded: {ARCFACE_MODEL}");
        Ok(Self { net, out_names })
    }

    /// Run ArcFace inference on a preprocessed blob and return an
    /// L2-normalized 512-d embedding vector.
    ///
    /// # Safety contract
    /// The output Mat is accessed via a bounds-checked raw pointer read.
    /// OpenCV guarantees 4-byte alignment for CV_32F Mats from its default
    /// allocator, and we verify the element count before reading.
    pub fn forward(&mut self, blob: &Mat) -> Result<Vec<f32>> {
        self.net.set_input(blob, "", 1.0, Scalar::default())?;
        let mut outputs: Vector<Mat> = Vector::new();
        self.net.forward(&mut outputs, &self.out_names)?;

        if outputs.is_empty() {
            return Err("ArcFace produced no output".into());
        }
        let out = outputs.get(0)?;

        // Validate shape: expect [1, 512] = 512 elements of CV_32F.
        if out.total() < EMBEDDING_DIM {
            return Err(format!(
                "ArcFace output too small: {} elements (expected {EMBEDDING_DIM})",
                out.total()
            ).into());
        }
        debug_assert!(
            out.is_continuous(),
            "ArcFace output Mat is not continuous — cannot safely read raw data"
        );

        // SAFETY: `out` is a continuous CV_32F Mat with at least EMBEDDING_DIM
        // elements.  OpenCV's default allocator guarantees 4-byte alignment for
        // f32 data.  The Mat outlives this slice access.
        let ptr = out.data() as *const f32;
        let raw: &[f32] = unsafe { std::slice::from_raw_parts(ptr, EMBEDDING_DIM) };

        // L2-normalize the raw embedding.
        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        Ok(raw.iter().map(|x| x / norm).collect())
    }
}

// ── Preprocessing ──────────────────────────────────────────────────────

/// Crop `rect` from `bgr_frame`, resize to 112×112, and convert to an
/// NCHW blob normalized with `(pixel - 127.5) / 128.0`.
pub fn preprocess(
    bgr_frame: &Mat,
    rect: Rect,
    resized: &mut Mat,
    normalized: &mut Mat,
) -> Result<()> {
    let roi = Mat::roi(bgr_frame, rect)?;
    imgproc::resize(
        &roi,
        resized,
        Size::new(ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    // blob_from_image_def creates NCHW [1, 3, 112, 112] from BGR.
    let blob = dnn::blob_from_image_def(resized)?;
    // Normalize: (pixel - 127.5) / 128.0
    blob.convert_to(normalized, core::CV_32F, 1.0 / 128.0, -127.5 / 128.0)?;
    Ok(())
}

// ── Similarity & averaging ─────────────────────────────────────────────

/// Compute cosine similarity between two L2-normalized embedding vectors.
///
/// Because both vectors are already unit-length, this is simply the dot
/// product.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum();
    sum as f64
}

/// Average multiple embeddings element-wise, then L2-normalize the result.
///
/// Returns an empty vector if `embeddings` is empty (no panic).
pub fn average_embeddings(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return vec![0.0f32; EMBEDDING_DIM];
    }

    let n = embeddings.len() as f32;
    let mut avg = vec![0.0f32; EMBEDDING_DIM];
    for emb in embeddings {
        for (a, v) in avg.iter_mut().zip(emb.iter()) {
            *a += v;
        }
    }
    for v in avg.iter_mut() {
        *v /= n;
    }
    let norm = avg.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    avg.iter().map(|x| x / norm).collect()
}

// ── Embedding database (JSON persistence) ──────────────────────────────

/// Maps identity name → averaged 512-d embedding vector.
pub type EmbeddingDb = HashMap<String, Vec<f32>>;

/// Load the embedding database from [`EMBEDDINGS_PATH`].
///
/// Returns an empty database if the file does not exist.
pub fn load_embeddings() -> Result<EmbeddingDb> {
    if !Path::new(EMBEDDINGS_PATH).exists() {
        return Ok(HashMap::new());
    }
    let data = fs::read_to_string(EMBEDDINGS_PATH)?;
    parse_json(&data)
}

/// Persist the embedding database to [`EMBEDDINGS_PATH`] as JSON.
///
/// Creates parent directories if they do not exist.  Identity names are
/// properly escaped so special characters (quotes, backslashes) do not
/// corrupt the file.
pub fn save_embeddings(db: &EmbeddingDb) -> Result<()> {
    fs::create_dir_all(Path::new(EMBEDDINGS_PATH).parent().unwrap())?;
    let mut f = fs::File::create(EMBEDDINGS_PATH)?;
    write!(f, "{{")?;
    let mut first = true;
    for (name, emb) in db {
        if !first { write!(f, ",")?; }
        first = false;
        write!(f, "\"{escaped}\":[")?;
        for (i, v) in emb.iter().enumerate() {
            if i > 0 { write!(f, ",")?; }
            write!(f, "{v:.6}")?;
        }
        write!(f, "]")?;
    }
    writeln!(f, "}}")?;
    Ok(())
}

/// control characters.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"'  => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                for unit in c.encode_utf16(&mut [0; 2]) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out
}

/// Unescape a JSON string value (handles `\"`, `\\`, `\n`, `\r`, `\t`).
fn unescape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('"')  => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n')  => out.push('\n'),
                Some('r')  => out.push('\r'),
                Some('t')  => out.push('\t'),
                Some('u')  => {
                    // \uXXXX — parse 4 hex digits.
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Ok(code) = u16::from_str_radix(&hex, 16) {
                        if let Some(c) = char::from_u32(code as u32) {
                            out.push(c);
                        }
                    }
                }
                Some(other) => { out.push('\\'); out.push(other); }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// Minimal JSON parser for the `{"name": [f32, ...], ...}` format used by
/// the embedding database.
///
/// This avoids a `serde_json` dependency for a single, well-defined schema.
/// The parser correctly handles:
/// - escaped characters in key names
/// - the trailing `]` on the last entry (no trailing comma)
/// - whitespace between tokens
fn parse_json(json: &str) -> Result<EmbeddingDb> {
    let mut db = EmbeddingDb::new();
    let json = json.trim();
    if json.is_empty() || json == "{}" {
        return Ok(db);
    }

    // Strip outer braces.
    let inner = json.strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or("Invalid embedding JSON: missing outer braces")?
        .trim();

    if inner.is_empty() {
        return Ok(db);
    }

    // State-machine approach: find each key-value pair by tracking bracket
    let bytes = inner.as_bytes();
    let len = bytes.len();
    let mut pos = 0;

    while pos < len {
        while pos < len && (bytes[pos] == b',' || bytes[pos].is_ascii_whitespace()) {
            pos += 1;
        }
        if pos >= len { break; }

        // Expect opening quote for key.
        if bytes[pos] != b'"' {
            return Err(format!(
                "Invalid embedding JSON at position {pos}: expected '\"', found '{}'",
                bytes[pos] as char
            ).into());
        }
        pos += 1; // skip opening "

        // Read key (handle escape sequences).
        let key_start = pos;
        let mut key = String::new();
        let mut found_end_quote = false;
        while pos < len {
            if bytes[pos] == b'\\' && pos + 1 < len {
                pos += 2; // skip escaped character
                continue;
            }
            if bytes[pos] == b'"' {
                key = unescape_json_string(&inner[key_start..pos]);
                pos += 1;
                found_end_quote = true;
                break;
            }
            pos += 1;
        }
        if !found_end_quote {
            return Err("Invalid embedding JSON: unterminated key string".into());
        }

        // Skip whitespace + colon.
        while pos < len && bytes[pos].is_ascii_whitespace() { pos += 1; }
        if pos >= len || bytes[pos] != b':' {
            return Err(format!("Invalid embedding JSON: expected ':' after key \"{key}\"").into());
        }
        pos += 1; // skip :

        // Skip whitespace + opening bracket.
        while pos < len && bytes[pos].is_ascii_whitespace() { pos += 1; }
        if pos >= len || bytes[pos] != b'[' {
            return Err(format!("Invalid embedding JSON: expected '[' for value of \"{key}\"").into());
        }
        pos += 1;

        // Read until matching ']'.
        let val_start = pos;
        let mut depth = 1;
        while pos < len && depth > 0 {
            match bytes[pos] {
                b'[' => depth += 1,
                b']' => depth -= 1,
                _ => {}
            }
            pos += 1;
        }
        if depth != 0 {
            return Err(format!("Invalid embedding JSON: unterminated array for \"{key}\"").into());
        }

        let vals_str = &inner[val_start..pos - 1];
        let vals: std::result::Result<Vec<f32>, _> = vals_str
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect();
        let vals = vals.map_err(|e| format!("Failed to parse embedding for \"{key}\": {e}"))?;

        if vals.len() != EMBEDDING_DIM {
            return Err(format!(
                "Embedding for \"{key}\" has {} values (expected {EMBEDDING_DIM})",
                vals.len()
            ).into());
        }

        db.insert(key, vals);
    }

    Ok(db)
}

/// Find the closest identity in the database by cosine similarity.
///
/// Returns `("UNKNOWN", -1.0)` if the database is empty.
pub fn find_best_match(db: &EmbeddingDb, query: &[f32]) -> (String, f64) {
    let mut best_name = "UNKNOWN".to_string();
    let mut best_sim = -1.0f64;
    for (name, stored) in db {
        let sim = cosine_similarity(query, stored);
        if sim > best_sim {
            best_sim = sim;
            best_name = name.clone();
        }
    }
    (best_name, best_sim)
}
