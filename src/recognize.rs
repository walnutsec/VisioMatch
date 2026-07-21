//! Live face recognition with stable track IDs and a heads-up display.
//!
//! Runs the camera loop, detects faces via the Haar cascade, computes
//! ArcFace embeddings, matches them against enrolled identities, and renders
//! a cyberpunk-style HUD with structural mesh, per-face track info, and an
//! FPS counter.

use opencv::{
    core::{AlgorithmHint, Mat, Point, Point2f, Rect, Scalar, Vector},
    features, highgui, imgproc,
    prelude::*,
};
use std::time::Instant;

use crate::{arcface, camera, detect, Result};

// ── Track state ────────────────────────────────────────────────────────

/// Persistent per-face track used to stabilize the HUD across frames.
///
/// Tracks are associated to detections via centroid distance (proportional
/// to the face's bounding-box size) and are dropped after [`TRACK_MAX_MISSED`]
/// consecutive frames without a match.
struct Track {
    /// Unique monotonic ID shown on the HUD.
    id: u32,
    /// Centroid of the last matched bounding box, in full-frame pixels.
    centroid: (f32, f32),
    /// Display name of the best-matching enrolled identity.
    name: String,
    /// EMA-smoothed cosine similarity for stable HUD readout.
    confidence: f64,
    /// Consecutive frames this track has not been associated with a detection.
    missed: u32,
}

/// Maximum centroid-distance multiplier relative to face diagonal for
/// track association.  A value of 0.6 means a detection can be up to 60%
/// of the face diagonal away and still match the same track.
const TRACK_DIST_RATIO: f32 = 0.6;

/// Number of consecutive missed frames before a track is dropped.
const TRACK_MAX_MISSED: u32 = 20;

/// EMA smoothing factor for the FPS counter.  Higher = more responsive
/// but noisier; lower = smoother but more latent.
const FPS_EMA_ALPHA: f64 = 0.15;

/// EMA weight for the new observation when smoothing track confidence.
const CONFIDENCE_EMA_NEW: f64 = 0.4;

// ── HUD colors (BGR) ──────────────────────────────────────────────────

const NEON_CYAN: Scalar = Scalar::new(255.0, 200.0, 0.0, 0.0);
const MESH_DIM: Scalar = Scalar::new(100.0, 50.0, 0.0, 0.0);
const NEON_WHITE: Scalar = Scalar::new(240.0, 240.0, 240.0, 0.0);
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

    let mut tracks: Vec<Track> = Vec::new();
    let mut next_track_id: u32 = 1;

    // Hoisted Mats — reused across frames to avoid per-frame heap churn.
    // OpenCV skips internal realloc when size/type already match.
    let mut frame = Mat::default();
    let mut gray = Mat::default();
    let mut processed = Mat::default();

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
        imgproc::equalize_hist(&gray.clone(), &mut gray)?;

        // Dim the live frame as the HUD backdrop.
        frame.convert_to(&mut processed, -1, 0.55, 0.0)?;

        let faces: Vector<Rect> = detect::detect_faces_scaled(&mut cascade, &gray)?;
        let max_w = frame.cols();
        let max_h = frame.rows();
        let mut seen_track_ids: Vec<u32> = Vec::new();

        for face in faces.iter() {
            let clamped = match detect::clamp_rect(face, max_w, max_h) {
                Some(r) => r,
                None => continue,
            };

            let centroid = (
                (clamped.x + clamped.width / 2) as f32,
                (clamped.y + clamped.height / 2) as f32,
            );

            // --- ArcFace identity match ---
            let blob = arcface::preprocess(&frame, clamped)?;
            let embedding = net.forward(&blob)?;
            let (best_name, similarity) = arcface::find_best_match(&db, &embedding);
            let is_match = similarity >= arcface::COSINE_THRESHOLD;
            let display_name = if is_match { best_name } else { "UNKNOWN".to_string() };

            // --- Match or create a stable track ---
            // B5: Distance threshold proportional to face diagonal, not a
            // fixed pixel count.  This handles varying camera distances and
            // frame resolutions without hard-coding.
            let face_diag = ((clamped.width * clamped.width + clamped.height * clamped.height) as f32).sqrt();
            let max_dist = face_diag * TRACK_DIST_RATIO;

            let track_idx = tracks.iter().position(|t| {
                let dx = t.centroid.0 - centroid.0;
                let dy = t.centroid.1 - centroid.1;
                (dx * dx + dy * dy).sqrt() < max_dist
            });

            let track_id = match track_idx {
                Some(idx) => {
                    let t = &mut tracks[idx];
                    t.centroid = centroid;
                    t.name = display_name.clone();
                    // EMA smoothing on cosine similarity.
                    t.confidence = t.confidence * (1.0 - CONFIDENCE_EMA_NEW) + similarity * CONFIDENCE_EMA_NEW;
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
                        confidence: similarity,
                        missed: 0,
                    });
                    id
                }
            };
            seen_track_ids.push(track_id);

            // --- Cosmetic structural mesh (Shi-Tomasi nodal features) ---
            draw_face_mesh(&gray, &mut processed, clamped, max_w, max_h)?;

            // --- HUD: corner brackets + identity info ---
            draw_face_hud(
                &mut processed,
                clamped,
                is_match,
                tracks.iter().find(|t| t.id == track_id).unwrap(),
                &display_name,
            )?;
        }

        // Drop tracks that have not been associated with a detection recently.
        for t in tracks.iter_mut() {
            if !seen_track_ids.contains(&t.id) {
                t.missed += 1;
            }
        }
        tracks.retain(|t| t.missed < TRACK_MAX_MISSED);

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

// ── Helper: structural mesh ────────────────────────────────────────────

/// Draw a Shi-Tomasi corner-based "wireframe mesh" inside the face bounding
/// box.  This is purely cosmetic — it gives the HUD a biometric-scan feel.
fn draw_face_mesh(
    gray: &Mat,
    canvas: &mut Mat,
    face_rect: Rect,
    max_w: i32,
    max_h: i32,
) -> crate::Result<()> {
    // Shrink the ROI by 10px on each side to avoid edge artifacts.
    let inner_raw = Rect::new(
        face_rect.x + 10,
        face_rect.y + 10,
        (face_rect.width - 20).max(1),
        (face_rect.height - 20).max(1),
    );
    let inner = detect::clamp_rect(inner_raw, max_w, max_h).unwrap_or(face_rect);
    let face_roi = Mat::roi(gray, inner)?;

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

    for (i, corner) in corners_vec.iter().enumerate() {
        let p1 = Point::new(
            corner.x as i32 + inner.x,
            corner.y as i32 + inner.y,
        );
        imgproc::circle(canvas, p1, 1, NEON_WHITE, -1, imgproc::LINE_AA, 0)?;
        for other in &corners_vec[i + 1..] {
            let p2 = Point::new(
                other.x as i32 + inner.x,
                other.y as i32 + inner.y,
            );
            let dx = p1.x - p2.x;
            let dy = p1.y - p2.y;
            if ((dx * dx + dy * dy) as f64).sqrt() < 45.0 {
                imgproc::line(canvas, p1, p2, MESH_DIM, 1, imgproc::LINE_AA, 0)?;
            }
        }
    }
    Ok(())
}

// ── Helper: per-face HUD ───────────────────────────────────────────────

/// Draw corner brackets and identity information next to the face bounding box.
fn draw_face_hud(
    canvas: &mut Mat,
    rect: Rect,
    is_match: bool,
    track: &Track,
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
        format!("TRACK  : #{:03}", track.id),
        format!("NAME   : {}", display_name),
        format!("SIM    : {:.2}", track.confidence),
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
