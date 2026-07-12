use opencv::{
    core::{self, AlgorithmHint, Mat, Point, Rect, Scalar, Size, Vector},
    dnn, geometry, highgui, imgproc,
    prelude::*,
};
use std::{
    path::Path,
    time::{Duration, Instant},
};

use crate::{camera, detect, Result};

pub fn run() -> Result<()> {
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

    let mut cap = camera::open()?;
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
            let hand_rect = match detect::clamp_rect(
                Rect::new(bbox.x - pad, bbox.y - pad, side + 2 * pad, side + 2 * pad),
                frame_w, frame_h,
            ) {
                Some(r) if r.width >= 50 && r.height >= 50 => r,
                _ => continue,
            };

            // ── Preprocess & run ONNX landmark model ─────────────────
            let hand_roi = Mat::roi(&flipped, hand_rect)?;

            // Model expects NHWC [1, 224, 224, 3].
            // blob_from_image creates NCHW [1, 3, 224, 224] — wrong shape.
            // Preprocess manually and copy into a 4D CV_32F mat.
            imgproc::resize(&hand_roi, &mut resized_hand, Size::new(224, 224), 0.0, 0.0, imgproc::INTER_LINEAR)?;
            imgproc::cvt_color(&resized_hand, &mut rgb_hand, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
            rgb_hand.convert_to(&mut float_hwc, core::CV_32FC3, 1.0 / 255.0, 0.0)?;

            // float_hwc is CV_32FC3 [224×224]: memory = R,G,B,R,G,B,... (HWC).
            // NHWC [1,224,224,3] is byte-for-byte identical with N=1 prepended.
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
