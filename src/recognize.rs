use opencv::{
    core::{AlgorithmHint, Mat, Point, Point2f, Rect, Scalar, Vector},
    features, highgui, imgproc,
    prelude::*,
};
use std::time::{Duration, Instant};

use crate::{arcface, camera, detect, Result};

struct Track {
    id: u32,
    centroid: (f32, f32),
    name: String,
    confidence: f64,
    missed: u32,
}

pub fn run() -> Result<()> {
    let db = arcface::load_embeddings()?;
    if db.is_empty() {
        eprintln!("[-] No enrolled identities found. Enroll someone first:");
        eprintln!("    cargo run -- enroll <name>");
        return Ok(());
    }

    let mut net = arcface::load_arcface()?;
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
            let embedding = arcface::forward(&mut net, &blob)?;
            let (best_name, similarity) = arcface::find_best_match(&db, &embedding);
            let is_match = similarity >= arcface::COSINE_THRESHOLD;
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
            let inner = detect::clamp_rect(inner_raw, max_w, max_h).unwrap_or(clamped);
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
            &format!("IDENTITIES: {}  //  THRESHOLD: {:.2}", db.len(), arcface::COSINE_THRESHOLD),
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
