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

// ── ArcFace constants ──────────────────────────────────────────────────
pub const ARCFACE_MODEL: &str = "data/models/w600k_mbf.onnx";
const ARCFACE_INPUT_SIZE: i32 = 112;
pub const EMBEDDING_DIM: usize = 512;
/// Cosine similarity threshold: faces above this value are considered a match.
/// ArcFace cosine similarities for same-person pairs typically land in 0.4–0.8;
/// cross-person pairs are usually below 0.3. Start with 0.4 and tune.
pub const COSINE_THRESHOLD: f64 = 0.4;
pub const EMBEDDINGS_PATH: &str = "data/embeddings.json";

// ── Model loading ──────────────────────────────────────────────────────

pub fn load_arcface() -> Result<dnn::Net> {
    if !Path::new(ARCFACE_MODEL).exists() {
        return Err(format!(
            "ArcFace model not found at: {ARCFACE_MODEL}\n\
             Download it:\n\
               wget -O data/models/w600k_mbf.onnx \\\n\
                 https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx"
        ).into());
    }
    let mut net = dnn::read_net_from_onnx_def(ARCFACE_MODEL)?;
    // Use OpenCL on the Intel UHD iGPU when available; falls back to CPU.
    net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
    net.set_preferable_target(dnn::DNN_TARGET_OPENCL)?;
    println!("[*] ArcFace model loaded: {ARCFACE_MODEL}");
    Ok(net)
}

// ── Preprocessing ──────────────────────────────────────────────────────

/// Preprocess a BGR face crop for ArcFace:
///   1. Crop the detection rect from the full-color frame
///   2. Resize to 112×112
///   3. Normalize: (pixel - 127.5) / 128.0
///   4. Create NCHW blob [1, 3, 112, 112]
pub fn preprocess(bgr_frame: &Mat, rect: Rect) -> Result<Mat> {
    let roi = Mat::roi(bgr_frame, rect)?;
    let mut resized = Mat::default();
    imgproc::resize(
        &roi,
        &mut resized,
        Size::new(ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;
    // blob_from_image_def creates NCHW [1, 3, 112, 112] from BGR.
    let blob = dnn::blob_from_image_def(&resized)?;
    // Normalize: (pixel - 127.5) / 128.0
    let mut normalized = Mat::default();
    blob.convert_to(&mut normalized, core::CV_32F, 1.0 / 128.0, -127.5 / 128.0)?;
    Ok(normalized)
}

// ── Inference ──────────────────────────────────────────────────────────

/// Run ArcFace inference on a preprocessed blob, return L2-normalized 512-d embedding.
pub fn forward(net: &mut dnn::Net, blob: &Mat) -> Result<Vec<f32>> {
    net.set_input(blob, "", 1.0, Scalar::default())?;
    let out_names = net.get_unconnected_out_layers_names()?;
    let mut outputs: Vector<Mat> = Vector::new();
    net.forward(&mut outputs, &out_names)?;
    if outputs.is_empty() {
        return Err("ArcFace produced no output".into());
    }
    let out = outputs.get(0)?;
    // out shape: [1, 512]
    if (out.total() as usize) < EMBEDDING_DIM {
        return Err(format!("ArcFace output too small: {} (expected {})", out.total(), EMBEDDING_DIM).into());
    }
    let ptr = out.data() as *const f32;
    let raw: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, EMBEDDING_DIM) }.to_vec();
    // L2-normalize
    let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    Ok(raw.iter().map(|x| x / norm).collect())
}

// ── Similarity & averaging ─────────────────────────────────────────────

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    // Both vectors are already L2-normalized, so dot product = cosine.
    a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum()
}

/// Average multiple embeddings element-wise, then L2-normalize.
pub fn average_embeddings(embeddings: &[Vec<f32>]) -> Vec<f32> {
    let n = embeddings.len() as f32;
    let mut avg = vec![0.0f32; EMBEDDING_DIM];
    for emb in embeddings {
        for (i, v) in emb.iter().enumerate() {
            avg[i] += v;
        }
    }
    for v in avg.iter_mut() {
        *v /= n;
    }
    let norm = avg.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    avg.iter().map(|x| x / norm).collect()
}

// ── Embedding database (JSON persistence) ──────────────────────────────

/// Stores name → averaged 512-d embedding vector.
pub type EmbeddingDb = HashMap<String, Vec<f32>>;

pub fn load_embeddings() -> Result<EmbeddingDb> {
    if !Path::new(EMBEDDINGS_PATH).exists() {
        return Ok(HashMap::new());
    }
    let data = fs::read_to_string(EMBEDDINGS_PATH)?;
    let db: EmbeddingDb = parse_json(&data)?;
    Ok(db)
}

pub fn save_embeddings(db: &EmbeddingDb) -> Result<()> {
    fs::create_dir_all(Path::new(EMBEDDINGS_PATH).parent().unwrap())?;
    let mut f = fs::File::create(EMBEDDINGS_PATH)?;
    write!(f, "{{")?;
    let mut first = true;
    for (name, emb) in db {
        if !first { write!(f, ",")?; }
        first = false;
        write!(f, "\"{}\":[{}]", name,
            emb.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(","))?;
    }
    writeln!(f, "}}")?;
    Ok(())
}

/// Minimal JSON parser for our simple {"name": [f32, ...], ...} format.
/// We avoid pulling in serde_json as a dependency for this one use case.
fn parse_json(json: &str) -> Result<EmbeddingDb> {
    let mut db = EmbeddingDb::new();
    let json = json.trim();
    if json.is_empty() || json == "{}" {
        return Ok(db);
    }
    // Strip outer braces
    let inner = json.strip_prefix('{').and_then(|s| s.strip_suffix('}'))
        .ok_or("Invalid embedding JSON: missing braces")?;
    // Split on "]," to get key-value pairs, but the last one won't have trailing comma
    for entry in inner.split("],") {
        let entry = entry.trim().trim_end_matches('}');
        if entry.is_empty() { continue; }
        let (key_part, vals_part) = entry.split_once(":[")
            .ok_or_else(|| format!("Invalid embedding entry: {}", &entry[..entry.len().min(60)]))?;
        let name = key_part.trim().trim_matches('"').to_string();
        let vals_str = vals_part.trim_end_matches(']');
        let vals: std::result::Result<Vec<f32>, _> = vals_str.split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect();
        let vals = vals.map_err(|e| format!("Failed to parse embedding for '{name}': {e}"))?;
        if vals.len() != EMBEDDING_DIM {
            return Err(format!("Embedding for '{name}' has {} values (expected {EMBEDDING_DIM})", vals.len()).into());
        }
        db.insert(name, vals);
    }
    Ok(db)
}

/// Find the closest identity in the database by cosine similarity.
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
