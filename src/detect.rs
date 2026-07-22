//! Haar cascade face detection with half-resolution scaling.
//!
//! Runs the OpenCV Haar cascade at [`DETECT_SCALE`] (50%) of the input
//! resolution to reduce detection latency, then maps bounding boxes back to
//! full-frame coordinates.

use opencv::{
    core::{Mat, Rect, Size, Vector},
    imgproc,
    prelude::*,
    xobjdetect,
};
use std::path::Path;

use crate::Result;

/// Path to the Haar frontal-face cascade model.
///
/// The local path (from `setup`) is preferred; system paths are fallbacks
/// for users who already have OpenCV data files installed.
pub const CASCADE_PATH: &str = "data/models/haarcascade_frontalface_default.xml";

/// Fallback paths searched if the local cascade file is not found.
const CASCADE_FALLBACKS: &[&str] = &[
    "/usr/share/opencv5/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
];

/// Detection runs at this fraction of the input resolution.
///
/// 0.5 halves each dimension (4× fewer pixels for the cascade to scan),
/// which roughly doubles detection throughput at the cost of missing very
/// small faces (below ~40px at the original resolution).
const DETECT_SCALE: f64 = 0.5;

/// Minimum face size **at the original resolution** (pixels).
///
/// Faces smaller than this are ignored.  The value is scaled down by
/// [`DETECT_SCALE`] before being passed to the cascade.
const CASCADE_MIN_FACE: i32 = 80;

/// Find the first existing cascade file path (local or system fallback).
///
/// Returns the path as a `String`, or an error if none exist.
pub fn find_cascade() -> Result<String> {
    if Path::new(CASCADE_PATH).exists() {
        return Ok(CASCADE_PATH.to_string());
    }
    for fallback in CASCADE_FALLBACKS {
        if Path::new(fallback).exists() {
            return Ok(fallback.to_string());
        }
    }
    Err(format!(
        "Haar cascade not found at {CASCADE_PATH} or system paths. Run `cargo run -- setup` to download it."
    ).into())
}

/// Load and return a [`CascadeClassifier`] from the first available cascade path.
pub fn load_cascade() -> Result<xobjdetect::CascadeClassifier> {
    let path = find_cascade()?;
    println!("[*] Using cascade: {path}");
    Ok(xobjdetect::CascadeClassifier::new(&path)?)
}

/// Clamp a rectangle to fit within `[0, max_w) × [0, max_h)`.
///
/// Returns `None` if the clamped rectangle has zero or negative area
/// (i.e., it was entirely outside the valid region).
pub fn clamp_rect(r: Rect, max_w: i32, max_h: i32) -> Option<Rect> {
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

/// Detect faces in a grayscale image using the Haar cascade at half resolution.
///
/// The returned rectangles are in the coordinate space of the **original**
/// (full-resolution) image.
pub fn detect_faces_scaled(
    cascade: &mut xobjdetect::CascadeClassifier,
    gray: &Mat,
    small: &mut Mat,
) -> Result<Vector<Rect>> {
    imgproc::resize(
        gray,
        small,
        Size::new(0, 0),
        DETECT_SCALE,
        DETECT_SCALE,
        imgproc::INTER_LINEAR,
    )?;

    let min_face = (CASCADE_MIN_FACE as f64 * DETECT_SCALE) as i32;
    let mut faces_small: Vector<Rect> = Vector::new();
    cascade.detect_multi_scale(
        small,
        &mut faces_small,
        1.1,
        5,
        0,
        Size::new(min_face, min_face),
        Size::new(0, 0),
    )?;

    // Map detections back to full-resolution coordinates.
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
