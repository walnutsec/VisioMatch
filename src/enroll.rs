use opencv::{
    core::{AlgorithmHint, Mat, Point, Rect, Scalar, Vector},
    highgui, imgcodecs, imgproc,
    prelude::*,
};
use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

use crate::{arcface, camera, detect, Result};

const ENROLL_TARGET: usize = 5;
const DATA_DIR: &str = "data/faces";

pub fn run(name: &str) -> Result<()> {
    let dir = Path::new(DATA_DIR).join(name);
    fs::create_dir_all(&dir)?;

    let mut net = arcface::load_arcface()?;
    let mut cascade = detect::load_cascade()?;
    let mut cap = camera::open()?;
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

        let faces: Vector<Rect> = detect::detect_faces_scaled(&mut cascade, &gray)?;

        let mut display = Mat::default();
        frame.copy_to(&mut display)?;

        if faces.len() == 1 {
            let face = faces.get(0)?;
            if let Some(clamped) = detect::clamp_rect(face, frame.cols(), frame.rows()) {
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
                    let blob = arcface::preprocess(&frame, clamped)?;
                    let embedding = arcface::forward(&mut net, &blob)?;
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
        let avg = arcface::average_embeddings(&embeddings);
        let mut db = arcface::load_embeddings()?;
        db.insert(name.to_string(), avg);
        arcface::save_embeddings(&db)?;
        println!(
            "[+] Enrolled '{}' ({} samples, {} total identities) -> {}",
            name, embeddings.len(), db.len(), arcface::EMBEDDINGS_PATH
        );
    } else {
        println!("[-] No samples captured — enrollment skipped.");
    }

    Ok(())
}
