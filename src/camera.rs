//! Camera setup and configuration.
//!
//! Opens the default V4L2 / platform camera at 720p @ 30 fps.

use opencv::{prelude::*, videoio};

use crate::Result;

/// Default camera device index (0 = first available camera).
const CAMERA_INDEX: i32 = 0;

/// Requested frame width in pixels.
const FRAME_WIDTH: f64 = 1280.0;

/// Requested frame height in pixels.
const FRAME_HEIGHT: f64 = 720.0;

/// Requested frames per second.
const FRAME_FPS: f64 = 30.0;

/// Open the default camera and configure it for 720p @ 30 fps.
///
/// The resolution and FPS are *requests* — the actual values depend on the
/// camera hardware.  Returns an error if the camera cannot be opened (e.g.,
/// device missing, insufficient permissions).
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
    Ok(cap)
}
