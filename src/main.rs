use opencv::{
    core::{self, Mat, Point, Point2f, Rect, Scalar, Size, Vector, AlgorithmHint},
    dnn,
    features,
    geometry,
    highgui, imgcodecs, imgproc,
    prelude::*,
    videoio,
    xobjdetect,
};
use std::{
    collections::HashMap,
    env, fs,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ── Face detection ─────────────────────────────────────────────────────
const CASCADE_PATHS: &[&str] = &[
    "/usr/share/opencv5/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
];
const DETECT_SCALE: f64 = 0.5;
const CASCADE_MIN_FACE: i32 = 80; // minimum face size in full-frame pixels

// ── ArcFace recognition ────────────────────────────────────────────────
const ARCFACE_MODEL: &str = "data/models/w600k_mbf.onnx";
const ARCFACE_INPUT_SIZE: i32 = 112;
const EMBEDDING_DIM: usize = 512;
/// Cosine similarity threshold: faces above this value are considered a match.
/// ArcFace cosine similarities for same-person pairs typically land in 0.4–0.8;
/// cross-person pairs are usually below 0.3. Start with 0.4 and tune.
const COSINE_THRESHOLD: f64 = 0.4;
/// Number of face samples captured during enrollment. More samples → more
/// robust average embedding. 5 is a good balance (vs LBPH's old 30).
const ENROLL_TARGET: usize = 5;
const EMBEDDINGS_PATH: &str = "data/embeddings.json";
const DATA_DIR: &str = "data/faces";

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("enroll") => {
            let name = args
                .get(2)
                .expect("Usage: cargo run -- enroll <name>");
            run_enroll(name)
        }
        Some("recognize") | None => run_recognize(),
        Some("hands") | Some("hand") => run_hands(),
        Some(other) => {
            eprintln!("[-] Unknown command: {other}");
            eprintln!("Usage:\n  cargo run -- enroll <name>\n  cargo run -- recognize\n  cargo run -- hand[s]");
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------

fn find_cascade() -> Result<&'static str> {
    for path in CASCADE_PATHS {
        if Path::new(path).exists() {
            return Ok(path);
        }
    }
    Err(format!(
        "Haar cascade not found. Searched:\n{}",
        CASCADE_PATHS.iter().map(|p| format!("  - {p}")).collect::<Vec<_>>().join("\n")
    ).into())
}

fn load_cascade() -> Result<xobjdetect::CascadeClassifier> {
    let path = find_cascade()?;
    println!("[*] Using cascade: {path}");
    Ok(xobjdetect::CascadeClassifier::new(path)?)
}

fn open_camera() -> Result<videoio::VideoCapture> {
    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cap)? {
        return Err("[-] FATAL: Could not open camera device (index 0). Check permissions or connection.".into());
    }
    cap.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0)?;
    cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0)?;
    cap.set(videoio::CAP_PROP_FPS, 30.0)?;
    Ok(cap)
}

fn clamp_rect(r: Rect, max_w: i32, max_h: i32) -> Option<Rect> {
    let x = r.x.max(0);
    let y = r.y.max(0);
    let w = (r.x + r.width).min(max_w) - x;
    let h = (r.y + r.height).min(max_h) - y;
    if w <= 0 || h <= 0 {
        None
    } else {
        Some(Rect::new(x, y, w, h))
    }
}

/// Detect faces on a downscaled copy of `gray` (DETECT_SCALE), then map the
/// resulting rectangles back up to full-frame coordinates. This is the FPS
/// fix: same effective minimum detectable face size as before, far fewer
/// pixels for the Haar cascade to scan.
fn detect_faces_scaled(
    cascade: &mut xobjdetect::CascadeClassifier,
    gray: &Mat,
) -> Result<Vector<Rect>> {
    let mut small = Mat::default();
    imgproc::resize(
        gray,
        &mut small,
        Size::new(0, 0),
        DETECT_SCALE,
        DETECT_SCALE,
        imgproc::INTER_LINEAR,
    )?;

    let min_face = (CASCADE_MIN_FACE as f64 * DETECT_SCALE) as i32;
    let mut faces_small: Vector<Rect> = Vector::new();
    cascade.detect_multi_scale(
        &small,
        &mut faces_small,
        1.1,
        5,
        0,
        Size::new(min_face, min_face),
        Size::new(0, 0),
    )?;

    let scale_back = 1.0 / DETECT_SCALE;
    let mut faces: Vector<Rect> = Vector::new();
    for r in faces_small.iter() {
        faces.push(Rect::new(
            (r.x as f64 * scale_back) as i32,
            (r.y as f64 * scale_back) as i32,
            (r.width as f64 * scale_back) as i32,
            (r.height as f64 * scale_back) as i32,
        ));
    }
    Ok(faces)
}

// ---------------------------------------------------------------------
// ArcFace embedding helpers
// ---------------------------------------------------------------------

fn load_arcface() -> Result<dnn::Net> {
    if !Path::new(ARCFACE_MODEL).exists() {
        return Err(format!(
            "ArcFace model not found at: {ARCFACE_MODEL}\n\
             Download it:\n\
               wget -O data/models/w600k_mbf.onnx \\
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

/// Preprocess a BGR face crop for ArcFace:
///   1. Crop the detection rect from the full-color frame
///   2. Resize to 112×112
///   3. Normalize: (pixel - 127.5) / 128.0
///   4. Create NCHW blob [1, 3, 112, 112]
fn arcface_preprocess(bgr_frame: &Mat, rect: Rect) -> Result<Mat> {
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
    // blob_from_image does: BGR→BGR (no swap), resize (already done),
    //   subtract mean 127.5, scale by 1/128 = 0.0078125, output NCHW.
    let blob = dnn::blob_from_image_def(
        &resized,
    )?;
    // Now apply normalization manually: (pixel - 127.5) / 128.0
    // blob_from_image_def doesn't normalize, so we do it ourselves.
    let mut normalized = Mat::default();
    blob.convert_to(&mut normalized, core::CV_32F, 1.0 / 128.0, -127.5 / 128.0)?;
    Ok(normalized)
}

/// Run ArcFace inference on a preprocessed blob, return L2-normalized 512-d embedding.
fn arcface_forward(net: &mut dnn::Net, blob: &Mat) -> Result<Vec<f32>> {
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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    // Both vectors are already L2-normalized, so dot product = cosine.
    a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum()
}

/// Average multiple embeddings element-wise, then L2-normalize.
fn average_embeddings(embeddings: &[Vec<f32>]) -> Vec<f32> {
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

// ---------------------------------------------------------------------
// Embedding database (JSON persistence)
// ---------------------------------------------------------------------

/// Stores name → averaged 512-d embedding vector.
type EmbeddingDb = HashMap<String, Vec<f32>>;

fn load_embeddings() -> Result<EmbeddingDb> {
    if !Path::new(EMBEDDINGS_PATH).exists() {
        return Ok(HashMap::new());
    }
    let data = fs::read_to_string(EMBEDDINGS_PATH)?;
    let db: EmbeddingDb = serde_json_minimal_parse(&data)?;
    Ok(db)
}

fn save_embeddings(db: &EmbeddingDb) -> Result<()> {
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
fn serde_json_minimal_parse(json: &str) -> Result<EmbeddingDb> {
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
fn find_best_match(db: &EmbeddingDb, query: &[f32]) -> (String, f64) {
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

// ---------------------------------------------------------------------
// Enrollment: capture face samples, extract & average embeddings
// ---------------------------------------------------------------------

fn run_enroll(name: &str) -> Result<()> {
    let dir = Path::new(DATA_DIR).join(name);
    fs::create_dir_all(&dir)?;

    let mut net = load_arcface()?;
    let mut cascade = load_cascade()?;
    let mut cap = open_camera()?;
    highgui::named_window("ENROLL", highgui::WINDOW_NORMAL)?;

    println!("[*] Enrolling '{name}'. Move your head slightly between captures.");
    println!("[*] Need {ENROLL_TARGET} samples. Press 'q' to cancel.");
    let mut last_capture = Instant::now() - Duration::from_secs(1);
    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let mut sample_idx = 0usize;

    while sample_idx < ENROLL_TARGET {
        let mut frame = Mat::default();
        cap.read(&mut frame)?;
        if frame.empty() {
            continue;
        }

        // Detection runs on grayscale (equalized)
        let mut gray = Mat::default();
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
        let mut eq = Mat::default();
        imgproc::equalize_hist(&gray, &mut eq)?;
        gray = eq;

        let faces: Vector<Rect> = detect_faces_scaled(&mut cascade, &gray)?;

        let mut display = Mat::default();
        frame.copy_to(&mut display)?;

        if faces.len() == 1 {
            let face = faces.get(0)?;
            if let Some(clamped) = clamp_rect(face, frame.cols(), frame.rows()) {
                imgproc::rectangle(
                    &mut display,
                    clamped,
                    Scalar::new(80.0, 255.0, 120.0, 0.0),
                    2,
                    imgproc::LINE_AA,
                    0,
                )?;

                if last_capture.elapsed() >= Duration::from_millis(500) {
                    // Save the face crop for reference
                    let roi = Mat::roi(&frame, clamped)?;
                    let path = dir.join(format!("{sample_idx:03}.png"));
                    imgcodecs::imwrite(path.to_str().unwrap(), &roi, &Vector::<i32>::new())?;

                    // Extract ArcFace embedding from the BGR frame
                    let blob = arcface_preprocess(&frame, clamped)?;
                    let embedding = arcface_forward(&mut net, &blob)?;
                    embeddings.push(embedding);

                    sample_idx += 1;
                    last_capture = Instant::now();
                    println!("[+] Captured {sample_idx}/{ENROLL_TARGET}");
                }
            }
        }

        imgproc::put_text(
            &mut display,
            &format!("Enrolling '{}': {}/{}", name, sample_idx, ENROLL_TARGET),
            Point::new(20, 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            Scalar::new(255.0, 200.0, 0.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;
        imgproc::put_text(
            &mut display,
            "Press 'q' to cancel",
            Point::new(20, 55),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::new(255.0, 200.0, 0.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;

        highgui::imshow("ENROLL", &display)?;
        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 {
            println!("[-] Enrollment stopped at {sample_idx}/{ENROLL_TARGET} samples.");
            break;
        }
    }

    highgui::destroy_window("ENROLL")?;

    if !embeddings.is_empty() {
        let avg = average_embeddings(&embeddings);
        let mut db = load_embeddings()?;
        db.insert(name.to_string(), avg);
        save_embeddings(&db)?;
        println!(
            "[+] Enrolled '{}' ({} samples, {} total identities) -> {EMBEDDINGS_PATH}",
            name, embeddings.len(), db.len()
        );
    } else {
        println!("[-] No samples captured — enrollment skipped.");
    }

    Ok(())
}

// ---------------------------------------------------------------------
// Recognition: live identification with persistent track IDs
// ---------------------------------------------------------------------

struct Track {
    id: u32,
    centroid: (f32, f32),
    name: String,
    confidence: f64,
    missed: u32,
}

fn run_recognize() -> Result<()> {
    let db = load_embeddings()?;
    if db.is_empty() {
        eprintln!("[-] No enrolled identities found. Enroll someone first:");
        eprintln!("    cargo run -- enroll <name>");
        return Ok(());
    }

    let mut net = load_arcface()?;
    let mut cascade = load_cascade()?;
    let mut cap = open_camera()?;

    highgui::named_window("BIOMETRIC_TOPOLOGY", highgui::WINDOW_NORMAL)?;
    highgui::set_window_property(
        "BIOMETRIC_TOPOLOGY",
        highgui::WND_PROP_FULLSCREEN,
        highgui::WINDOW_FULLSCREEN as f64,
    )?;

    println!(
        "[*] ARCFACE RECOGNITION ENGINE ONLINE // {} known identities loaded",
        db.len()
    );

    let mut fps_timer = Instant::now();
    let mut frame_count = 0;
    let mut fps = 0.0;

    let neon_cyan = Scalar::new(255.0, 200.0, 0.0, 0.0);
    let mesh_dim = Scalar::new(100.0, 50.0, 0.0, 0.0);
    let neon_white = Scalar::new(240.0, 240.0, 240.0, 0.0);
    let color_match = Scalar::new(80.0, 255.0, 120.0, 0.0);
    let color_unknown = Scalar::new(60.0, 60.0, 255.0, 0.0);

    let mut tracks: Vec<Track> = Vec::new();
    let mut next_track_id: u32 = 1;

    loop {
        let mut frame = Mat::default();
        cap.read(&mut frame)?;
        if frame.empty() {
            continue;
        }

        frame_count += 1;
        if fps_timer.elapsed() >= Duration::from_secs(1) {
            fps = frame_count as f64;
            frame_count = 0;
            fps_timer = Instant::now();
        }

        // Detection runs on equalized grayscale
        let mut gray = Mat::default();
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
        let mut eq = Mat::default();
        imgproc::equalize_hist(&gray, &mut eq)?;
        gray = eq;

        // Dim the live frame as the HUD backdrop
        let mut processed = Mat::default();
        frame.convert_to(&mut processed, -1, 0.55, 0.0)?;

        let faces: Vector<Rect> = detect_faces_scaled(&mut cascade, &gray)?;
        let max_w = frame.cols();
        let max_h = frame.rows();
        let mut seen_track_ids: Vec<u32> = Vec::new();

        for face in faces.iter() {
            let clamped = match clamp_rect(face, max_w, max_h) {
                Some(r) => r,
                None => continue,
            };

            let centroid = (
                (clamped.x + clamped.width / 2) as f32,
                (clamped.y + clamped.height / 2) as f32,
            );

            // --- ArcFace identity match ---
            let blob = arcface_preprocess(&frame, clamped)?;
            let embedding = arcface_forward(&mut net, &blob)?;
            let (best_name, similarity) = find_best_match(&db, &embedding);
            let is_match = similarity >= COSINE_THRESHOLD;
            let display_name = if is_match { best_name } else { "UNKNOWN".to_string() };
            let confidence = similarity;

            // --- match/create a stable track so the HUD doesn't flicker ---
            let track_idx = tracks.iter().position(|t| {
                let dx = t.centroid.0 - centroid.0;
                let dy = t.centroid.1 - centroid.1;
                (dx * dx + dy * dy).sqrt() < 80.0
            });

            let track_id = match track_idx {
                Some(idx) => {
                    let t = &mut tracks[idx];
                    t.centroid = centroid;
                    t.name = display_name.clone();
                    // EMA smoothing on cosine similarity
                    t.confidence = t.confidence * 0.6 + confidence * 0.4;
                    t.missed = 0;
                    t.id
                }
                None => {
                    let id = next_track_id;
                    next_track_id += 1;
                    tracks.push(Track {
                        id,
                        centroid,
                        name: display_name.clone(),
                        confidence,
                        missed: 0,
                    });
                    id
                }
            };
            seen_track_ids.push(track_id);

            // --- cosmetic structural mesh (Shi-Tomasi nodal features) ---
            let inner_raw = Rect::new(
                clamped.x + 10,
                clamped.y + 10,
                (clamped.width - 20).max(1),
                (clamped.height - 20).max(1),
            );
            let inner = clamp_rect(inner_raw, max_w, max_h).unwrap_or(clamped);
            let face_roi = Mat::roi(&gray, inner)?;

            let mut corners = Vector::<Point2f>::new();
            features::good_features_to_track(
                &face_roi,
                &mut corners,
                45,
                0.02,
                12.0,
                &Mat::default(),
                3,
                false,
                0.04,
            )?;
            let corners_vec: Vec<Point2f> = corners.iter().collect();

            for i in 0..corners_vec.len() {
                let p1 = Point::new(
                    corners_vec[i].x as i32 + inner.x,
                    corners_vec[i].y as i32 + inner.y,
                );
                imgproc::circle(&mut processed, p1, 1, neon_white, -1, imgproc::LINE_AA, 0)?;
                for j in (i + 1)..corners_vec.len() {
                    let p2 = Point::new(
                        corners_vec[j].x as i32 + inner.x,
                        corners_vec[j].y as i32 + inner.y,
                    );
                    let dx = p1.x - p2.x;
                    let dy = p1.y - p2.y;
                    if ((dx * dx + dy * dy) as f64).sqrt() < 45.0 {
                        imgproc::line(&mut processed, p1, p2, mesh_dim, 1, imgproc::LINE_AA, 0)?;
                    }
                }
            }

            // --- HUD ---
            let accent = if is_match { color_match } else { color_unknown };
            let (x, y, w, h) = (clamped.x, clamped.y, clamped.width, clamped.height);
            let len = 30;
            imgproc::line(&mut processed, Point::new(x, y), Point::new(x + len, y), accent, 1, imgproc::LINE_AA, 0)?;
            imgproc::line(&mut processed, Point::new(x, y), Point::new(x, y + len), accent, 1, imgproc::LINE_AA, 0)?;
            imgproc::line(&mut processed, Point::new(x, y + h), Point::new(x + len, y + h), accent, 1, imgproc::LINE_AA, 0)?;
            imgproc::line(&mut processed, Point::new(x, y + h), Point::new(x, y + h - len), accent, 1, imgproc::LINE_AA, 0)?;

            let track = tracks.iter().find(|t| t.id == track_id).unwrap();
            let lines = [
                format!("TRACK  : #{:03}", track.id),
                format!("NAME   : {}", track.name),
                format!("SIM    : {:.2}", track.confidence),
                format!("NODES  : {:02}", corners_vec.len()),
                format!("STATUS : {}", if is_match { "MATCH" } else { "UNKNOWN" }),
            ];
            for (i, line) in lines.iter().enumerate() {
                imgproc::put_text(
                    &mut processed,
                    line,
                    Point::new(x + w + 15, y + 18 + (i as i32) * 18),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.4,
                    accent,
                    1,
                    imgproc::LINE_AA,
                    false,
                )?;
            }
        }

        // drop tracks that haven't been seen in a while
        for t in tracks.iter_mut() {
            if !seen_track_ids.contains(&t.id) {
                t.missed += 1;
            }
        }
        tracks.retain(|t| t.missed < 20);

        imgproc::put_text(
            &mut processed,
            &format!("FPS: {fps:.1}  //  ARCFACE"),
            Point::new(20, 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.45,
            neon_cyan,
            1,
            imgproc::LINE_AA,
            false,
        )?;
        imgproc::put_text(
            &mut processed,
            &format!("IDENTITIES: {}  //  THRESHOLD: {COSINE_THRESHOLD:.2}", db.len()),
            Point::new(20, 55),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.45,
            neon_cyan,
            1,
            imgproc::LINE_AA,
            false,
        )?;

        highgui::imshow("BIOMETRIC_TOPOLOGY", &processed)?;
        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 {
            break;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------
// Hand tracking: skin-color ROI detection + MediaPipe landmark ONNX model
// ---------------------------------------------------------------------

fn run_hands() -> Result<()> {
    const HAND_MODEL: &str =
        "data/models/handpose_estimation_mediapipe_2023feb.onnx";

    if !Path::new(HAND_MODEL).exists() {
        eprintln!("[!] Hand landmark model not found at: {HAND_MODEL}");
        eprintln!("    Download it:");
        eprintln!("      mkdir -p data/models && cd data/models");
        eprintln!("      wget https://github.com/opencv/opencv_zoo/raw/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx");
        eprintln!("    Then: cargo run -- hands");
        return Ok(());
    }

    let mut net = dnn::read_net_from_onnx_def(HAND_MODEL)?;
    // DNN_BACKEND_OPENCV + DNN_TARGET_OPENCL offloads inference to the
    // Intel UHD G4 iGPU (48 EUs) via OpenCL. Requires intel-compute-runtime.
    // Falls back to CPU transparently if no OpenCL device is available.
    net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
    net.set_preferable_target(dnn::DNN_TARGET_OPENCL)?;

    let out_names = net.get_unconnected_out_layers_names()?;

    let mut cap = open_camera()?;
    highgui::named_window("HAND_TOPOLOGY", highgui::WINDOW_NORMAL)?;
    highgui::set_window_property(
        "HAND_TOPOLOGY",
        highgui::WND_PROP_FULLSCREEN,
        highgui::WINDOW_FULLSCREEN as f64,
    )?;

    // MediaPipe 21-keypoint hand skeleton
    // 0=Wrist, 1-4=Thumb, 5-8=Index, 9-12=Middle, 13-16=Ring, 17-20=Pinky
    const CONNECTIONS: &[(usize, usize)] = &[
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17), // wrist → finger bases
        (1, 2), (2, 3), (3, 4),                    // thumb
        (5, 6), (6, 7), (7, 8),                    // index
        (9, 10), (10, 11), (11, 12),               // middle
        (13, 14), (14, 15), (15, 16),              // ring
        (17, 18), (18, 19), (19, 20),              // pinky
        (5, 9), (9, 13), (13, 17),                 // palm arch
    ];

    let neon_cyan    = Scalar::new(255.0, 220.0, 0.0, 0.0);
    let left_color   = Scalar::new(255.0, 100.0, 0.0, 0.0);   // cyan tint
    let right_color  = Scalar::new(60.0, 255.0, 120.0, 0.0);  // green tint
    let joint_color  = Scalar::new(220.0, 220.0, 220.0, 0.0);

    // Morphology kernel built once, reused every frame
    let morph_kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(5, 5),
        Point::new(-1, -1),
    )?;

    let mut fps_timer = Instant::now();
    let mut frame_count = 0u32;
    let mut fps = 0.0f64;

    // Preprocessing scale — skin detection runs at half resolution
    const PREPROC_SCALE: f64 = 0.5;

    // Hoisted Mats — reused across frames to avoid per-frame heap churn.
    // OpenCV skips internal realloc when size/type already match.
    let mut frame = Mat::default();
    let mut flipped = Mat::default();
    let mut display = Mat::default();
    let mut small = Mat::default();
    let mut ycrcb = Mat::default();
    let mut skin_mask = Mat::default();
    let mut opened = Mat::default();
    let mut closed = Mat::default();
    let mut contours: Vector<Vector<Point>> = Vector::new();
    let mut resized_hand = Mat::default();
    let mut rgb_hand = Mat::default();
    let mut float_hwc = Mat::default();

    println!("[*] HAND TOPOLOGY ONLINE // {HAND_MODEL} loaded");

    loop {
        cap.read(&mut frame)?;
        if frame.empty() { continue; }

        frame_count += 1;
        if fps_timer.elapsed() >= Duration::from_secs(1) {
            fps = frame_count as f64;
            frame_count = 0;
            fps_timer = Instant::now();
        }

        // Flip horizontally so handedness agrees with mirror-selfie convention
        core::flip(&frame, &mut flipped, 1)?;

        // Dim backdrop for HUD readability
        flipped.convert_to(&mut display, -1, 0.65, 0.0)?;

        let frame_w = flipped.cols();
        let frame_h = flipped.rows();

        // Downscale for skin detection — morphology runs on 25% of the pixels
        imgproc::resize(
            &flipped, &mut small,
            Size::new(0, 0), PREPROC_SCALE, PREPROC_SCALE,
            imgproc::INTER_LINEAR,
        )?;

        // ── Skin-color mask (YCrCb space) on downscaled frame ─────────
        imgproc::cvt_color(
            &small, &mut ycrcb,
            imgproc::COLOR_BGR2YCrCb, 0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // Y: any, Cr: 133-173, Cb: 77-127 — standard robust skin range
        core::in_range(
            &ycrcb,
            &Scalar::new(0.0, 133.0, 77.0, 0.0),
            &Scalar::new(255.0, 173.0, 127.0, 0.0),
            &mut skin_mask,
        )?;

        // Open → remove speckle noise; Close → fill gaps in palm/fingers
        // 5×5 kernel × 1 iteration is sufficient for bounding-rect detection
        let border_val = Scalar::all(f64::MAX); // morphologyDefaultBorderValue()
        imgproc::morphology_ex(
            &skin_mask, &mut opened, imgproc::MORPH_OPEN, &morph_kernel,
            Point::new(-1, -1), 1, core::BORDER_CONSTANT, border_val,
        )?;
        imgproc::morphology_ex(
            &opened, &mut closed, imgproc::MORPH_CLOSE, &morph_kernel,
            Point::new(-1, -1), 1, core::BORDER_CONSTANT, border_val,
        )?;

        // ── Find hand candidates from contours (downscaled mask) ──────
        contours.clear();
        imgproc::find_contours(
            &closed,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        // Sort by area descending, keep at most 2 (a human has two hands)
        let scale_back = 1.0 / PREPROC_SCALE;
        let min_area_small = 6000.0 * PREPROC_SCALE * PREPROC_SCALE;
        let mut candidates: Vec<(usize, f64)> = Vec::new();
        for i in 0..contours.len() {
            let area = geometry::contour_area(&contours.get(i)?, false)?;
            if area >= min_area_small {
                candidates.push((i, area));
            }
        }
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(2);

        for &(ci, _) in &candidates {
            let contour = contours.get(ci)?;

            // Scale bounding rect back to full-resolution coordinates
            let bbox_small = geometry::bounding_rect(&contour)?;
            let bbox = Rect::new(
                (bbox_small.x as f64 * scale_back) as i32,
                (bbox_small.y as f64 * scale_back) as i32,
                (bbox_small.width as f64 * scale_back) as i32,
                (bbox_small.height as f64 * scale_back) as i32,
            );

            // Square-pad around the bounding box for a stable crop
            let side = bbox.width.max(bbox.height);
            let pad  = side / 3;
            let hand_rect = match clamp_rect(
                Rect::new(bbox.x - pad, bbox.y - pad, side + 2 * pad, side + 2 * pad),
                frame_w, frame_h,
            ) {
                Some(r) if r.width >= 50 && r.height >= 50 => r,
                _ => continue,
            };

            // ── Preprocess & run ONNX landmark model ─────────────────
            let hand_roi = Mat::roi(&flipped, hand_rect)?;

            // Model expects NHWC [1, 224, 224, 3].
            // blob_from_image creates NCHW [1, 3, 224, 224] — confirmed wrong by the
            // error: OpenCV DNN received [1, 224, 3, 224] (shape mismatch symptom).
            // Preprocess manually and copy into a 4D CV_32F mat with the right shape.
            imgproc::resize(&hand_roi, &mut resized_hand, Size::new(224, 224), 0.0, 0.0, imgproc::INTER_LINEAR)?;
            imgproc::cvt_color(&resized_hand, &mut rgb_hand, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
            rgb_hand.convert_to(&mut float_hwc, core::CV_32FC3, 1.0 / 255.0, 0.0)?;

            // float_hwc is CV_32FC3 [224×224]: memory = R,G,B,R,G,B,... (HWC).
            // NHWC [1,224,224,3] is byte-for-byte identical with N=1 prepended.
            // Copy into a 4D single-channel mat so OpenCV DNN sees the right shape.
            let nhwc_sizes = [1i32, 224, 224, 3];
            let n_floats = 224 * 224 * 3_usize;
            let blob = unsafe {
                let mut b = Mat::new_nd(&nhwc_sizes, core::CV_32F)?;
                std::ptr::copy_nonoverlapping(
                    float_hwc.data() as *const f32,
                    b.data_mut() as *mut f32,
                    n_floats,
                );
                b
            };
            net.set_input(&blob, "", 1.0, Scalar::default())?;

            let mut outputs: Vector<Mat> = Vector::new();
            // NOTE: if this line doesn't compile, try net.forward_1(&mut outputs, &out_names)?
            net.forward(&mut outputs, &out_names)?;

            if outputs.is_empty() { continue; }

            // ── Parse output 0: 63 floats = 21 landmarks × {x, y, z} ──
            let lm_mat = outputs.get(0)?;
            if lm_mat.total() < 63 { continue; }
            let lm_ptr = lm_mat.data() as *const f32;
            // SAFETY: lm_mat owns the data, is alive for this scope, CV_32F
            let lm_slice = unsafe { std::slice::from_raw_parts(lm_ptr, 63) };

            // ── Parse output 1: handedness score ───────────────────────
            // Score in [0, 1]: >0.5 = Right hand.
            // After our horizontal flip, "Right" model output = right hand
            // in real world (mirror convention corrected).
            let raw_hand_score = if outputs.len() > 1 {
                let h = outputs.get(1)?;
                if h.total() >= 1 { unsafe { *(h.data() as *const f32) } } else { 0.5 }
            } else { 0.5 };
            // Defensively apply sigmoid in case the model outputs a raw logit
            let hand_score = if raw_hand_score > 1.0 || raw_hand_score < 0.0 {
                1.0 / (1.0 + (-raw_hand_score).exp())
            } else { raw_hand_score };

            let is_right   = hand_score > 0.5;
            let hand_label = if is_right { "RIGHT" } else { "LEFT" };
            let accent     = if is_right { right_color } else { left_color };

            // ── Parse output 2: hand presence confidence ───────────────
            let presence = if outputs.len() > 2 {
                let p = outputs.get(2)?;
                if p.total() >= 1 {
                    let raw = unsafe { *(p.data() as *const f32) };
                    // Presence is often a raw logit from MediaPipe ONNX exports
                    1.0 / (1.0 + (-raw).exp())
                } else { 1.0f32 }
            } else { 1.0f32 };

            if presence < 0.5 { continue; } // model says no hand here

            // ── Project landmarks to frame coordinates ─────────────────
            // Landmarks are normalized [0,1] relative to the 224×224 crop.
            let cx = hand_rect.x as f32;
            let cy = hand_rect.y as f32;
            let cw = hand_rect.width as f32;
            let ch = hand_rect.height as f32;

            let mut pts = [(0i32, 0i32); 21];
            for j in 0..21 {
                pts[j] = (
                    (lm_slice[j * 3]     * cw + cx) as i32,
                    (lm_slice[j * 3 + 1] * ch + cy) as i32,
                );
            }

            // ── Draw skeleton ──────────────────────────────────────────
            for &(a, b) in CONNECTIONS {
                imgproc::line(
                    &mut display,
                    Point::new(pts[a].0, pts[a].1),
                    Point::new(pts[b].0, pts[b].1),
                    accent, 1, imgproc::LINE_AA, 0,
                )?;
            }
            for &(px, py) in &pts {
                imgproc::circle(
                    &mut display, Point::new(px, py),
                    3, joint_color, -1, imgproc::LINE_AA, 0,
                )?;
            }

            // ── HUD: corner bracket + label ───────────────────────────
            let (bx, by, bw, bh) = (hand_rect.x, hand_rect.y, hand_rect.width, hand_rect.height);
            let tick = 18i32;
            imgproc::line(&mut display, Point::new(bx, by),           Point::new(bx + tick, by),           accent, 1, imgproc::LINE_AA, 0)?;
            imgproc::line(&mut display, Point::new(bx, by),           Point::new(bx, by + tick),           accent, 1, imgproc::LINE_AA, 0)?;
            imgproc::line(&mut display, Point::new(bx + bw, by),      Point::new(bx + bw - tick, by),      accent, 1, imgproc::LINE_AA, 0)?;
            imgproc::line(&mut display, Point::new(bx + bw, by),      Point::new(bx + bw, by + tick),      accent, 1, imgproc::LINE_AA, 0)?;
            imgproc::line(&mut display, Point::new(bx, by + bh),      Point::new(bx + tick, by + bh),      accent, 1, imgproc::LINE_AA, 0)?;
            imgproc::line(&mut display, Point::new(bx, by + bh),      Point::new(bx, by + bh - tick),      accent, 1, imgproc::LINE_AA, 0)?;
            imgproc::line(&mut display, Point::new(bx + bw, by + bh), Point::new(bx + bw - tick, by + bh), accent, 1, imgproc::LINE_AA, 0)?;
            imgproc::line(&mut display, Point::new(bx + bw, by + bh), Point::new(bx + bw, by + bh - tick), accent, 1, imgproc::LINE_AA, 0)?;

            let hud = [
                format!("{hand_label} HAND"),
                format!("SIDE : {:.0}%", hand_score * 100.0),
                format!("CONF : {:.0}%", presence  * 100.0),
            ];
            for (idx, line) in hud.iter().enumerate() {
                imgproc::put_text(
                    &mut display, line,
                    Point::new(bx, by - 8 - (hud.len() as i32 - 1 - idx as i32) * 16),
                    imgproc::FONT_HERSHEY_SIMPLEX, 0.42,
                    accent, 1, imgproc::LINE_AA, false,
                )?;
            }
        }

        // ── Global HUD ────────────────────────────────────────────────
        imgproc::put_text(
            &mut display, &format!("FPS: {fps:.1}"),
            Point::new(20, 30), imgproc::FONT_HERSHEY_SIMPLEX,
            0.45, neon_cyan, 1, imgproc::LINE_AA, false,
        )?;
        imgproc::put_text(
            &mut display, "HAND TOPOLOGY  //  'q' to quit",
            Point::new(20, 55), imgproc::FONT_HERSHEY_SIMPLEX,
            0.45, neon_cyan, 1, imgproc::LINE_AA, false,
        )?;

        highgui::imshow("HAND_TOPOLOGY", &display)?;
        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 { break; }
    }

    Ok(())
}
