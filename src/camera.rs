//! Camera setup and configuration.

use opencv::{prelude::*, videoio};

use crate::Result;

/// Default camera device
const CAMERA_INDEX: i32 = 0;

const FRAME_WIDTH: f64 = 1280.0;

const FRAME_HEIGHT: f64 = 720.0;

const FRAME_FPS: f64 = 30.0;

pub fn open() -> Result<videoio::VideoCapture> {
    let mut cap = videoio::VideoCapture::new(CAMERA_INDEX, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cap)? {
        return Err(format!(
            "[-] FATAL: Could not open camera device (index {CAMERA_INDEX}). \
             Check permissions or connection."
        ).into());
    }
    cap.set(videoio::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)?;
    cap.set(videoio::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)?;
    cap.set(videoio::CAP_PROP_FPS, FRAME_FPS)?;
    let actual_w = cap.get(videoio::CAP_PROP_FRAME_WIDTH)?;
    let actual_h = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)?;
    if (actual_w - FRAME_WIDTH).abs() > 1.0 || (actual_h - FRAME_HEIGHT).abs() > 1.0 {
        println!("[!] Warning: Requested {}x{}, but camera provided {}x{}", 
            FRAME_WIDTH, FRAME_HEIGHT, actual_w, actual_h);
    }

    Ok(cap)
}
