use opencv::{prelude::*, videoio};

use crate::Result;

pub fn open() -> Result<videoio::VideoCapture> {
    let mut cap = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cap)? {
        return Err("[-] FATAL: Could not open camera device (index 0). Check permissions or connection.".into());
    }
    cap.set(videoio::CAP_PROP_FRAME_WIDTH, 1280.0)?;
    cap.set(videoio::CAP_PROP_FRAME_HEIGHT, 720.0)?;
    cap.set(videoio::CAP_PROP_FPS, 30.0)?;
    Ok(cap)
}
