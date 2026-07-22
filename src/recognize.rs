//! Live face recognition with stable track IDs and a heads-up display.
//!
//! Runs the camera loop, detects faces via the Haar cascade, computes
//! ArcFace embeddings, matches them against enrolled identities, and renders
//! a cyberpunk-style HUD with structural mesh, per-face track info, and an
//! FPS counter.

use opencv::{
    core::{AlgorithmHint, Mat, Point, Rect, Scalar, Vector},
    highgui, imgproc,
    prelude::*,
};
use std::time::Instant;

use crate::{arcface, camera, detect, tracker::{CentroidTracker, Detection}, Result};

// ── Track state ────────────────────────────────────────────────────────

/// EMA smoothing factor for the FPS counter.  Higher = more responsive
/// but noisier; lower = smoother but more latent.
const FPS_EMA_ALPHA: f64 = 0.15;

// ── HUD colors (BGR) ──────────────────────────────────────────────────

const NEON_CYAN: Scalar = Scalar::new(255.0, 200.0, 0.0, 0.0);
const COLOR_MATCH: Scalar = Scalar::new(80.0, 255.0, 120.0, 0.0);
const COLOR_UNKNOWN: Scalar = Scalar::new(60.0, 60.0, 255.0, 0.0);

/// Corner bracket half-length (pixels) drawn at bounding-box corners.
const BRACKET_LEN: i32 = 30;

// ── Entry point ────────────────────────────────────────────────────────

/// Run the live recognition loop.
///
/// Loads the embedding database and ArcFace model, opens the camera, and
/// enters a display loop that:
/// 1. Detects faces (Haar cascade at half resolution)
/// 2. Runs ArcFace inference to get a 512-d embedding per face
/// 3. Matches against enrolled identities by cosine similarity
/// 4. Maintains stable per-face tracks (centroid tracker + EMA smoothing)
/// 5. Renders a HUD with identity info, structural mesh, and FPS
///
/// Press `q` or `Esc` to exit.
pub fn run() -> Result<()> {
    let db = arcface::load_embeddings()?;
    if db.is_empty() {
        eprintln!("[-] No enrolled identities found. Enroll someone first:");
        eprintln!("    cargo run -- enroll <name>");
        return Ok(());
    }

    let mut net = arcface::ArcFaceNet::load()?;
    let mut cascade = detect::load_cascade()?;
    let mut cap = camera::open()?;

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

    // EMA-smoothed FPS counter.
    let mut fps = 0.0f64;
    let mut last_frame_time = Instant::now();

    let mut tracker = CentroidTracker::new();

    // Hoisted Mats — reused across frames to avoid per-frame heap churn.
    // OpenCV skips internal realloc when size/type already match.
    let mut frame = Mat::default();
    let mut gray = Mat::default();
    let mut processed = Mat::default();
    let mut small = Mat::default();
    let mut equalized = Mat::default();

    // Reusable Mats for ArcFace preprocessing
    let mut resized_face = Mat::default();
    let mut normalized_face = Mat::default();

    loop {
        cap.read(&mut frame)?;
        if frame.empty() {
            continue;
        }

        // EMA-based FPS: smoother and more accurate than reset-every-second.
        let now = Instant::now();
        let dt = now.duration_since(last_frame_time).as_secs_f64();
        last_frame_time = now;
        if dt > 0.0 {
            let instant_fps = 1.0 / dt;
            fps = fps * (1.0 - FPS_EMA_ALPHA) + instant_fps * FPS_EMA_ALPHA;
        }

        // Detection runs on equalized grayscale.
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
        imgproc::equalize_hist(&gray, &mut equalized)?;

        // Dim the live frame as the HUD backdrop.
        frame.convert_to(&mut processed, -1, 0.55, 0.0)?;

        let faces: Vector<Rect> = detect::detect_faces_scaled(&mut cascade, &equalized, &mut small)?;
        let max_w = frame.cols();
        let max_h = frame.rows();
        let mut frame_detections = Vec::new();
        let mut face_rects = Vec::new();

        for face in faces.iter() {
            let clamped = match detect::clamp_rect(face, max_w, max_h) {
                Some(r) => r,
                None => continue,
            };

            // --- ArcFace identity match ---
            arcface::preprocess(&frame, clamped, &mut resized_face, &mut normalized_face)?;
            let embedding = net.forward(&normalized_face)?;
            let (best_name, similarity) = arcface::find_best_match(&db, &embedding);
            let is_match = similarity >= arcface::COSINE_THRESHOLD;
            let display_name = if is_match { best_name } else { "UNKNOWN".to_string() };

            frame_detections.push(Detection {
                cx: (clamped.x + clamped.width / 2) as f32,
                cy: (clamped.y + clamped.height / 2) as f32,
                name: display_name,
                confidence: similarity,
                face_diag: ((clamped.width * clamped.width + clamped.height * clamped.height) as f32).sqrt(),
            });
            face_rects.push((clamped, is_match));
        }

        let updated_tracks = tracker.update(&frame_detections);

        for (i, (clamped, is_match)) in face_rects.iter().enumerate() {
            if let Some((track_id, ref display_name, confidence)) = updated_tracks.get(i) {
                // --- HUD: corner brackets + identity info ---
                draw_face_hud(
                    &mut processed,
                    *clamped,
                    *is_match,
                    *track_id,
                    *confidence,
                    display_name,
                )?;
            }
        }

        // --- Global HUD ---
        imgproc::put_text(
            &mut processed,
            &format!("FPS: {fps:.1}  //  ARCFACE"),
            Point::new(20, 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.45,
            NEON_CYAN,
            1,
            imgproc::LINE_AA,
            false,
        )?;
        imgproc::put_text(
            &mut processed,
            &format!("IDENTITIES: {}  //  THRESHOLD: {:.2}", db.len(), arcface::COSINE_THRESHOLD),
            Point::new(20, 55),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.45,
            NEON_CYAN,
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



// ── Helper: per-face HUD

/// Draw corner brackets and identity information next to the face bounding box.
fn draw_face_hud(
    canvas: &mut Mat,
    rect: Rect,
    is_match: bool,
    track_id: u32,
    confidence: f64,
    display_name: &str,
) -> crate::Result<()> {
    let accent = if is_match { COLOR_MATCH } else { COLOR_UNKNOWN };
    let (x, y, w, h) = (rect.x, rect.y, rect.width, rect.height);

    // Corner brackets.
    imgproc::line(canvas, Point::new(x, y), Point::new(x + BRACKET_LEN, y), accent, 1, imgproc::LINE_AA, 0)?;
    imgproc::line(canvas, Point::new(x, y), Point::new(x, y + BRACKET_LEN), accent, 1, imgproc::LINE_AA, 0)?;
    imgproc::line(canvas, Point::new(x, y + h), Point::new(x + BRACKET_LEN, y + h), accent, 1, imgproc::LINE_AA, 0)?;
    imgproc::line(canvas, Point::new(x, y + h), Point::new(x, y + h - BRACKET_LEN), accent, 1, imgproc::LINE_AA, 0)?;

    // Identity info panel.
    let lines = [
        format!("TRACK  : #{:03}", track_id),
        format!("NAME   : {}", display_name),
        format!("SIM    : {:.2}", confidence),
        format!("STATUS : {}", if is_match { "MATCH" } else { "UNKNOWN" }),
    ];
    for (i, line) in lines.iter().enumerate() {
        imgproc::put_text(
            canvas,
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
    Ok(())
}
