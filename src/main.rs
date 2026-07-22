//! VisioMatch — ArcFace-powered real-time face recognition and hand tracking.
//!
//! CLI entry point that multiplexes the four operational modes:
//! - `setup`         — interactive first-time setup wizard
//! - `enroll <name>` — capture face samples and store averaged embeddings
//! - `recognize`     — live camera recognition against enrolled identities
//! - `hand[s]`       — MediaPipe hand landmark tracking

mod arcface;
mod camera;
mod detect;
mod enroll;
mod hands;
mod recognize;
mod setup;
mod tracker;

use std::{env, process};

/// Project-wide error type
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() {
    if let Err(e) = run() {
        eprintln!("[FATAL] {e}");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("enroll") => {
            let name = args
                .get(2)
                .expect("Usage: cargo run -- enroll <name>");
            enroll::run(name)
        }
        Some("recognize") | None => recognize::run(),
        Some("hands") | Some("hand") => hands::run(),
        Some("setup") | Some("install") => setup::run(),
        Some(other) => {
            eprintln!("[-] Unknown command: {other}");
            eprintln!(
                "Usage:\n  cargo run -- setup            Interactive first-time setup\n  cargo run -- enroll <name>    Enroll a face\n  cargo run -- recognize       Live recognition\n  cargo run -- hand[s]         Hand tracking"
            );
            process::exit(1);
        }
    }
}
