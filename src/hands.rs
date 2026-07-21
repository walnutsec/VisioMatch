//! Hand landmark tracking using a MediaPipe ONNX model.
//!
//! Detects up to two hands via skin-color segmentation (YCrCb), then runs
//! a 21-keypoint landmark model on each candidate region.  The output is
//! rendered as a skeleton overlay with handedness classification and a
//! corner-bracket HUD.

use opencv::{
    core::{self, AlgorithmHint, Mat, Point, Rect, Scalar, Size, Vector},
    dnn, geometry, highgui, imgproc,
    prelude::*,
};
use std::{
    path::Path,
    time::Instant,
};

use crate::{camera, detect, Result};

// ── Constants ──────────────────────────────────────────────────────────

/// Path to the MediaPipe hand landmark ONNX model (relative to project root).
const HAND_MODEL: &str = "data/models/handpose_estimation_mediapipe_2023feb.onnx";

/// The model expects a 224×224 RGB input.
const MODEL_INPUT_SIZE: i32 = 224;

/// Number of keypoints in the MediaPipe hand skeleton.
const NUM_LANDMARKS: usize = 21;

/// Skin detection and morphology run at this fraction of the input resolution.
///
/// 0.5 means 25% of pixels are processed (half in each dimension), which is
/// sufficient for bounding-box-level hand detection.
const PREPROC_SCALE: f64 = 0.5;

/// Minimum contour area **at the original resolution** (pixels²) for a region
/// to be considered a hand candidate.
const MIN_HAND_AREA: f64 = 6000.0;

/// Minimum hand presence confidence (after sigmoid) to accept a detection.
const MIN_PRESENCE: f32 = 0.5;

/// EMA smoothing factor for the FPS counter.
const FPS_EMA_ALPHA: f64 = 0.15;

// ── MediaPipe skeleton topology ────────────────────────────────────────

/// Bone connections for the 21-keypoint MediaPipe hand skeleton.
///
/// 0 = Wrist, 1–4 = Thumb, 5–8 = Index, 9–12 = Middle, 13–16 = Ring,
/// 17–20 = Pinky.
const CONNECTIONS: &[(usize, usize)] = &[
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17), // wrist → finger bases
    (1, 2), (2, 3), (3, 4),                    // thumb
    (5, 6), (6, 7), (7, 8),                    // index
    (9, 10), (10, 11), (11, 12),               // middle
    (13, 14), (14, 15), (15, 16),              // ring
    (17, 18), (18, 19), (19, 20),              // pinky
    (5, 9), (9, 13), (13, 17),                 // palm arch
];

// ── HUD colors (BGR) ──────────────────────────────────────────────────

const NEON_CYAN: Scalar = Scalar::new(255.0, 220.0, 0.0, 0.0);
const LEFT_COLOR: Scalar = Scalar::new(255.0, 100.0, 0.0, 0.0);
const RIGHT_COLOR: Scalar = Scalar::new(60.0, 255.0, 120.0, 0.0);
const JOINT_COLOR: Scalar = Scalar::new(220.0, 220.0, 220.0, 0.0);

/// Corner bracket half-length (pixels).
const BRACKET_TICK: i32 = 18;

// ── Entry point ────────────────────────────────────────────────────────

/// Run the hand-tracking loop.
///
/// Opens the camera, detects hand regions via skin-color segmentation,
/// runs the MediaPipe landmark model, and renders a skeleton overlay with
/// handedness and confidence.  Press `q` or `Esc` to exit.
pub fn run() -> Result<()> {
    if !Path::new(HAND_MODEL).exists() {
        eprintln!("[!] Hand landmark model not found at: {HAND_MODEL}");
        eprintln!("    Download it:");
        eprintln!("      mkdir -p data/models && cd data/models");
        eprintln!("      wget https://github.com/opencv/opencv_zoo/raw/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx");
        eprintln!("    Then: cargo run -- hands");
        return Ok(());
    }

    let mut net = dnn::read_net_from_onnx_def(HAND_MODEL)?;
    net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
    net.set_preferable_target(dnn::DNN_TARGET_OPENCL)?;

    // Cache output-layer names once (same rationale as ArcFaceNet).
    let out_names = net.get_unconnected_out_layers_names()?;

    let mut cap = camera::open()?;
    highgui::named_window("HAND_TOPOLOGY", highgui::WINDOW_NORMAL)?;
    highgui::set_window_property(
        "HAND_TOPOLOGY",
        highgui::WND_PROP_FULLSCREEN,
        highgui::WINDOW_FULLSCREEN as f64,
    )?;

    // Morphology kernel built once, reused every frame.
    let morph_kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(5, 5),
        Point::new(-1, -1),
    )?;

    // EMA-smoothed FPS counter.
    let mut fps = 0.0f64;
    let mut last_frame_time = Instant::now();

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

        // EMA-based FPS.
        let now = Instant::now();
        let dt = now.duration_since(last_frame_time).as_secs_f64();
        last_frame_time = now;
        if dt > 0.0 {
            let instant_fps = 1.0 / dt;
            fps = fps * (1.0 - FPS_EMA_ALPHA) + instant_fps * FPS_EMA_ALPHA;
        }

        // Flip horizontally so handedness agrees with mirror-selfie convention.
        core::flip(&frame, &mut flipped, 1)?;

        // Dim backdrop for HUD readability.
        flipped.convert_to(&mut display, -1, 0.65, 0.0)?;

        let frame_w = flipped.cols();
        let frame_h = flipped.rows();

        // Find hand candidate bounding boxes from skin-color segmentation.
        let candidates = find_hand_candidates(
            &flipped, &mut small, &mut ycrcb, &mut skin_mask,
            &mut opened, &mut closed, &mut contours, &morph_kernel,
        )?;

        for bbox in &candidates {
            // Square-pad around the bounding box for a stable crop.
            let side = bbox.width.max(bbox.height);
            let pad  = side / 3;
            let hand_rect = match detect::clamp_rect(
                Rect::new(bbox.x - pad, bbox.y - pad, side + 2 * pad, side + 2 * pad),
                frame_w, frame_h,
            ) {
                Some(r) if r.width >= 50 && r.height >= 50 => r,
                _ => continue,
            };

            // Preprocess and run the landmark model.
            let blob = preprocess_hand_blob(
                &flipped, hand_rect,
                &mut resized_hand, &mut rgb_hand, &mut float_hwc,
            )?;
            net.set_input(&blob, "", 1.0, Scalar::default())?;

            let mut outputs: Vector<Mat> = Vector::new();
            net.forward(&mut outputs, &out_names)?;
            if outputs.is_empty() { continue; }

            // Parse model outputs.
            let landmarks = match parse_landmarks(&outputs)? {
                Some(lm) => lm,
                None => continue,
            };
            let (hand_score, is_right) = parse_handedness(&outputs);
            let presence = parse_presence(&outputs);
            if presence < MIN_PRESENCE { continue; }

            // Project landmarks to frame coordinates.
            let pts = project_landmarks(&landmarks, hand_rect);

            // Render.
            let accent = if is_right { RIGHT_COLOR } else { LEFT_COLOR };
            let hand_label = if is_right { "RIGHT" } else { "LEFT" };
            draw_skeleton(&mut display, &pts, accent)?;
            draw_hand_hud(&mut display, hand_rect, hand_label, hand_score, presence, accent)?;
        }

        // Global HUD.
        imgproc::put_text(
            &mut display, &format!("FPS: {fps:.1}"),
            Point::new(20, 30), imgproc::FONT_HERSHEY_SIMPLEX,
            0.45, NEON_CYAN, 1, imgproc::LINE_AA, false,
        )?;
        imgproc::put_text(
            &mut display, "HAND TOPOLOGY  //  'q' to quit",
            Point::new(20, 55), imgproc::FONT_HERSHEY_SIMPLEX,
            0.45, NEON_CYAN, 1, imgproc::LINE_AA, false,
        )?;

        highgui::imshow("HAND_TOPOLOGY", &display)?;
        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 { break; }
    }

    Ok(())
}

// ── Skin-color segmentation ────────────────────────────────────────────

/// Detect hand candidate bounding boxes via YCrCb skin-color segmentation.
///
/// Operates at [`PREPROC_SCALE`] resolution for speed, then scales the
/// resulting bounding rectangles back to full-frame coordinates.  Returns
/// at most 2 candidates (one per hand), sorted by area descending.
#[allow(clippy::too_many_arguments)] // All params are pre-allocated buffer reuse.
fn find_hand_candidates(
    bgr_frame: &Mat,
    small: &mut Mat,
    ycrcb: &mut Mat,
    skin_mask: &mut Mat,
    opened: &mut Mat,
    closed: &mut Mat,
    contours: &mut Vector<Vector<Point>>,
    morph_kernel: &Mat,
) -> Result<Vec<Rect>> {
    // Downscale for skin detection — morphology runs on 25% of the pixels.
    imgproc::resize(
        bgr_frame, small,
        Size::new(0, 0), PREPROC_SCALE, PREPROC_SCALE,
        imgproc::INTER_LINEAR,
    )?;

    // YCrCb skin-color mask.  Y: any, Cr: 133–173, Cb: 77–127.
    imgproc::cvt_color(
        small, ycrcb,
        imgproc::COLOR_BGR2YCrCb, 0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    core::in_range(
        ycrcb,
        &Scalar::new(0.0, 133.0, 77.0, 0.0),
        &Scalar::new(255.0, 173.0, 127.0, 0.0),
        skin_mask,
    )?;

    // Open → remove speckle noise; Close → fill gaps in palm/fingers.
    let border_val = imgproc::morphology_default_border_value()?;
    imgproc::morphology_ex(
        skin_mask, opened, imgproc::MORPH_OPEN, morph_kernel,
        Point::new(-1, -1), 1, core::BORDER_CONSTANT, border_val,
    )?;
    imgproc::morphology_ex(
        opened, closed, imgproc::MORPH_CLOSE, morph_kernel,
        Point::new(-1, -1), 1, core::BORDER_CONSTANT, border_val,
    )?;

    // Find contours and select the two largest above the area threshold.
    contours.clear();
    imgproc::find_contours(
        closed,
        contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let scale_back = 1.0 / PREPROC_SCALE;
    let min_area_small = MIN_HAND_AREA * PREPROC_SCALE * PREPROC_SCALE;
    let mut candidates: Vec<(usize, f64)> = Vec::new();

    for i in 0..contours.len() {
        let area = geometry::contour_area(&contours.get(i)?, false)?;
        if area >= min_area_small {
            candidates.push((i, area));
        }
    }
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    candidates.truncate(2);

    // Scale bounding rects back to full-resolution coordinates.
    let mut result = Vec::with_capacity(candidates.len());
    for &(ci, _) in &candidates {
        let contour = contours.get(ci)?;
        let bbox_small = geometry::bounding_rect(&contour)?;
        result.push(Rect::new(
            (bbox_small.x as f64 * scale_back) as i32,
            (bbox_small.y as f64 * scale_back) as i32,
            (bbox_small.width as f64 * scale_back) as i32,
            (bbox_small.height as f64 * scale_back) as i32,
        ));
    }
    Ok(result)
}

// ── Preprocessing ──────────────────────────────────────────────────────

/// Prepare a hand crop for the MediaPipe ONNX model.
///
/// The model expects NHWC `[1, 224, 224, 3]` (RGB, float32, 0–1 range).
/// OpenCV's `blob_from_image` creates NCHW, so we preprocess manually:
/// resize → BGR→RGB → float normalize → reshape to 4D NHWC.
fn preprocess_hand_blob(
    bgr_frame: &Mat,
    hand_rect: Rect,
    resized: &mut Mat,
    rgb: &mut Mat,
    float_hwc: &mut Mat,
) -> Result<Mat> {
    let hand_roi = Mat::roi(bgr_frame, hand_rect)?;
    imgproc::resize(
        &hand_roi, resized,
        Size::new(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        0.0, 0.0, imgproc::INTER_LINEAR,
    )?;
    imgproc::cvt_color(resized, rgb, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
    rgb.convert_to(float_hwc, core::CV_32FC3, 1.0 / 255.0, 0.0)?;

    // float_hwc is CV_32FC3 [224×224]: memory layout is R,G,B,R,G,B,... (HWC).
    // NHWC [1,224,224,3] is byte-for-byte identical with N=1 prepended.
    let nhwc_sizes = [1i32, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3];
    let n_floats = (MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3) as usize;
    let blob = unsafe {
        let mut b = Mat::new_nd(&nhwc_sizes, core::CV_32F)?;
        debug_assert!(
            b.is_continuous(),
            "NHWC blob Mat is not continuous — layout assumption violated"
        );
        std::ptr::copy_nonoverlapping(
            float_hwc.data() as *const f32,
            b.data_mut() as *mut f32,
            n_floats,
        );
        b
    };
    Ok(blob)
}

// ── Output parsing ─────────────────────────────────────────────────────

/// Parse 21 hand landmarks from the model's first output tensor.
///
/// Each landmark is `(x, y, z)` normalized to `[0, 1]` relative to the
/// 224×224 input crop.  Returns `None` if the output is too small.
fn parse_landmarks(outputs: &Vector<Mat>) -> Result<Option<Vec<(f32, f32, f32)>>> {
    let lm_mat = outputs.get(0)?;
    let expected = NUM_LANDMARKS * 3;
    if lm_mat.total() < expected {
        return Ok(None);
    }

    // SAFETY: lm_mat owns the data, is alive for this scope, and is CV_32F
    // with at least `expected` elements.
    let lm_ptr = lm_mat.data() as *const f32;
    let lm_slice = unsafe { std::slice::from_raw_parts(lm_ptr, expected) };

    let mut landmarks = Vec::with_capacity(NUM_LANDMARKS);
    for j in 0..NUM_LANDMARKS {
        landmarks.push((
            lm_slice[j * 3],
            lm_slice[j * 3 + 1],
            lm_slice[j * 3 + 2],
        ));
    }
    Ok(Some(landmarks))
}

/// Parse handedness from the model's second output tensor.
///
/// Returns `(score, is_right)`.  Score is in `[0, 1]`; >0.5 = right hand.
/// If the output contains a raw logit (outside 0–1), sigmoid is applied.
/// After the horizontal flip, "Right" from the model = right in the real
/// world (mirror convention).
fn parse_handedness(outputs: &Vector<Mat>) -> (f32, bool) {
    let raw = if outputs.len() > 1 {
        if let Ok(h) = outputs.get(1) {
            if h.total() >= 1 { unsafe { *(h.data() as *const f32) } } else { 0.5 }
        } else { 0.5 }
    } else { 0.5 };

    // Apply sigmoid if the value looks like a raw logit.
    let score = if !(0.0..=1.0).contains(&raw) {
        1.0 / (1.0 + (-raw).exp())
    } else {
        raw
    };
    (score, score > 0.5)
}

/// Parse hand presence confidence from the model's third output tensor.
///
/// Applies sigmoid since MediaPipe ONNX exports often output raw logits.
fn parse_presence(outputs: &Vector<Mat>) -> f32 {
    if outputs.len() > 2 {
        if let Ok(p) = outputs.get(2) {
            if p.total() >= 1 {
                let raw = unsafe { *(p.data() as *const f32) };
                return 1.0 / (1.0 + (-raw).exp());
            }
        }
    }
    1.0
}

/// Project normalized landmark coordinates onto full-frame pixel coordinates.
fn project_landmarks(
    landmarks: &[(f32, f32, f32)],
    hand_rect: Rect,
) -> [(i32, i32); NUM_LANDMARKS] {
    let cx = hand_rect.x as f32;
    let cy = hand_rect.y as f32;
    let cw = hand_rect.width as f32;
    let ch = hand_rect.height as f32;

    let mut pts = [(0i32, 0i32); NUM_LANDMARKS];
    for (j, &(lx, ly, _)) in landmarks.iter().enumerate() {
        pts[j] = (
            (lx * cw + cx) as i32,
            (ly * ch + cy) as i32,
        );
    }
    pts
}

// ── Rendering ──────────────────────────────────────────────────────────

/// Draw the hand skeleton (bones + joints) onto the display canvas.
fn draw_skeleton(
    canvas: &mut Mat,
    pts: &[(i32, i32); NUM_LANDMARKS],
    accent: Scalar,
) -> Result<()> {
    for &(a, b) in CONNECTIONS {
        imgproc::line(
            canvas,
            Point::new(pts[a].0, pts[a].1),
            Point::new(pts[b].0, pts[b].1),
            accent, 1, imgproc::LINE_AA, 0,
        )?;
    }
    for &(px, py) in pts {
        imgproc::circle(
            canvas, Point::new(px, py),
            3, JOINT_COLOR, -1, imgproc::LINE_AA, 0,
        )?;
    }
    Ok(())
}

/// Draw corner brackets and the hand info HUD above the bounding box.
fn draw_hand_hud(
    canvas: &mut Mat,
    rect: Rect,
    label: &str,
    hand_score: f32,
    presence: f32,
    accent: Scalar,
) -> Result<()> {
    let (bx, by, bw, bh) = (rect.x, rect.y, rect.width, rect.height);

    // Corner brackets (all four corners).
    let corners = [
        (bx, by, bx + BRACKET_TICK, by, bx, by + BRACKET_TICK),
        (bx + bw, by, bx + bw - BRACKET_TICK, by, bx + bw, by + BRACKET_TICK),
        (bx, by + bh, bx + BRACKET_TICK, by + bh, bx, by + bh - BRACKET_TICK),
        (bx + bw, by + bh, bx + bw - BRACKET_TICK, by + bh, bx + bw, by + bh - BRACKET_TICK),
    ];
    for &(cx, cy, hx, hy, vx, vy) in &corners {
        imgproc::line(canvas, Point::new(cx, cy), Point::new(hx, hy), accent, 1, imgproc::LINE_AA, 0)?;
        imgproc::line(canvas, Point::new(cx, cy), Point::new(vx, vy), accent, 1, imgproc::LINE_AA, 0)?;
    }

    // Info labels above the bounding box.
    let hud = [
        format!("{label} HAND"),
        format!("SIDE : {:.0}%", hand_score * 100.0),
        format!("CONF : {:.0}%", presence * 100.0),
    ];
    for (idx, line) in hud.iter().enumerate() {
        imgproc::put_text(
            canvas, line,
            Point::new(bx, by - 8 - (hud.len() as i32 - 1 - idx as i32) * 16),
            imgproc::FONT_HERSHEY_SIMPLEX, 0.42,
            accent, 1, imgproc::LINE_AA, false,
        )?;
    }
    Ok(())
}
