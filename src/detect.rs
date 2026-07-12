use opencv::{
    core::{Mat, Point, Rect, Size, Vector},
    imgproc,
    prelude::*,
    xobjdetect,
};
use std::path::Path;

use crate::Result;

const CASCADE_PATHS: &[&str] = &[
    "/usr/share/opencv5/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
];
/// Haar detection runs on a downscaled image for speed. This is the
/// single biggest FPS win: same effective minimum face size, 75% fewer
/// pixels for the cascade to scan.
const DETECT_SCALE: f64 = 0.5;
const CASCADE_MIN_FACE: i32 = 80; // minimum face size in full-frame pixels

pub fn find_cascade() -> Result<&'static str> {
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

pub fn load_cascade() -> Result<xobjdetect::CascadeClassifier> {
    let path = find_cascade()?;
    println!("[*] Using cascade: {path}");
    Ok(xobjdetect::CascadeClassifier::new(path)?)
}

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

/// Detect faces on a downscaled copy of `gray` (DETECT_SCALE), then map the
/// resulting rectangles back up to full-frame coordinates.
pub fn detect_faces_scaled(
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
