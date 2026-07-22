//! Interactive first-time setup wizard.

use std::{
    fs,
    io::{self, Write},
    path::Path,
    process::Command,
};

use crate::{arcface, detect, Result};

// ── Model definitions

/// Descriptor for a downloadable model file.
struct ModelInfo {
    /// Human-readable name shown in the setup output.
    name: &'static str,
    /// Filesystem path
    path: &'static str,
    /// Download URL.
    url: &'static str,
    /// Approximate file size for display.
    size_hint: &'static str,
    /// Whether this model is required or optional.
    required: bool,
    /// Description shown to the user.
    description: &'static str,
}

const MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "ArcFace MobileFaceNet",
        path: "data/models/w600k_mbf.onnx",
        url: "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx",
        size_hint: "~13 MB",
        required: true,
        description: "Required for face recognition (enroll + recognize modes)",
    },
    ModelInfo {
        name: "MediaPipe Hand Landmarks",
        path: "data/models/handpose_estimation_mediapipe_2023feb.onnx",
        url: "https://github.com/opencv/opencv_zoo/raw/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx",
        size_hint: "~5 MB",
        required: false,
        description: "Optional — needed only for hand tracking mode",
    },
    ModelInfo {
        name: "Haar Cascade Frontal Face",
        path: "data/models/haarcascade_frontalface_default.xml",
        url: "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        size_hint: "~900 KB",
        required: true,
        description: "Required for face detection",
    },
];

// ── UI helpers

mod color {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const RED: &str = "\x1b[31m";
    pub const CYAN: &str = "\x1b[36m";
    pub const DIM: &str = "\x1b[2m";
}

fn ok(msg: &str) {
    println!("      {}{} ✓{} {msg}", color::BOLD, color::GREEN, color::RESET);
}

fn warn(msg: &str) {
    println!("      {}{} !{} {msg}", color::BOLD, color::YELLOW, color::RESET);
}

fn fail(msg: &str) {
    println!("      {}{} ✗{} {msg}", color::BOLD, color::RED, color::RESET);
}

fn step(num: usize, total: usize, title: &str) {
    println!(
        "\n  {}[{num}/{total}]{} {}{title}{}",
        color::CYAN, color::RESET, color::BOLD, color::RESET
    );
}

fn prompt_yn(question: &str, default: bool) -> bool {
    let hint = if default { "Y/n" } else { "y/N" };
    print!("      {question} [{hint}]: ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        return default;
    }
    match input.trim().to_lowercase().as_str() {
        "y" | "yes" => true,
        "n" | "no" => false,
        "" => default,
        _ => default,
    }
}

// ── Download logic

/// Detect which download tool is available
fn find_downloader() -> Option<&'static str> {
    ["curl", "wget"].iter()
        .find(|cmd| Command::new("which").arg(cmd).output().is_ok_and(|o| o.status.success()))
        .copied()
}

fn download_file(url: &str, dest: &str, tool: &str) -> Result<()> {
    if let Some(parent) = Path::new(dest).parent() {
        fs::create_dir_all(parent)?;
    }

    print!("      Downloading... ");
    io::stdout().flush()?;

    let status = match tool {
        "curl" => Command::new("curl")
            .args(["-L", "--progress-bar", "-o", dest, url])
            .status()?,
        "wget" => Command::new("wget")
            .args(["-q", "--show-progress", "-O", dest, url])
            .status()?,
        _ => return Err("No supported download tool found".into()),
    };

    if !status.success() {
        let _ = fs::remove_file(dest);
        return Err(format!("Download failed (exit code: {})", status).into());
    }

    Ok(())
}

// ── Main setup

pub fn run() -> Result<()> {
    println!();
    println!("  {}╔══════════════════════════════════════════════════════════════╗{}", color::CYAN, color::RESET);
    println!("  {}║{}                        VisioMatch                        {}║{}", color::CYAN, color::RESET, color::CYAN, color::RESET);
    println!("  {}╚══════════════════════════════════════════════════════════════╝{}", color::CYAN, color::RESET);

    let total_steps = 2 + MODELS.len();
    let mut current_step = 0;
    let mut had_errors = false;

    // ── Step 1
    current_step += 1;
    step(current_step, total_steps, "Checking download tools");

    let downloader = match find_downloader() {
        Some(tool) => {
            ok(&format!("Found: {tool}"));
            tool
        }
        None => {
            fail("Neither curl nor wget found");
            println!("      {}Install one of them:{}", color::DIM, color::RESET);
            println!("        sudo pacman -S curl    {}# Arch{}", color::DIM, color::RESET);
            println!("        sudo apt install curl  {}# Ubuntu/Debian{}", color::DIM, color::RESET);
            return Err("No download tool available. Install curl or wget.".into());
        }
    };

    // ── Step 2
    current_step += 1;
    step(current_step, total_steps, "Checking directory structure");

    fs::create_dir_all("data/models")?;
    fs::create_dir_all("data/faces")?;
    ok("data/models/ ready");
    ok("data/faces/ ready");

    // ── Steps 3
    for model in MODELS {
        current_step += 1;
        step(
            current_step,
            total_steps,
            &format!("{} ({})", model.name, model.size_hint),
        );
        println!(
            "      {}{}{}", color::DIM, model.description, color::RESET
        );

        if Path::new(model.path).exists() {
            let size = fs::metadata(model.path).map(|m| m.len()).unwrap_or(0);
            if size > 1024 {
                ok(&format!("Already downloaded ({:.1} MB)", size as f64 / 1_048_576.0));
                continue;
            } else {
                warn("File exists but looks truncated — re-downloading");
                let _ = fs::remove_file(model.path);
            }
        }

        if model.required {
            println!("      {}This model is {}required{}{} — downloading now.",
                color::DIM, color::YELLOW, color::DIM, color::RESET);
        } else {
            if !prompt_yn("Download this optional model?", true) {
                warn("Skipped (you can download it later with `cargo run -- setup`)");
                continue;
            }
        }

        match download_file(model.url, model.path, downloader) {
            Ok(()) => {
                let size = fs::metadata(model.path).map(|m| m.len()).unwrap_or(0);
                ok(&format!("Downloaded ({:.1} MB)", size as f64 / 1_048_576.0));
            }
            Err(e) => {
                fail(&format!("Download failed: {e}"));
                if model.required {
                    had_errors = true;
                }
            }
        }
    }

    // ── Verify Haar cascade ────────────────────────────────────
    println!(
        "\n  {}[+]{} {}Verifying Haar cascade{}",
        color::CYAN, color::RESET, color::BOLD, color::RESET
    );
    match detect::find_cascade() {
        Ok(path) => ok(&format!("Found: {path}")),
        Err(_) => {
            fail("Haar cascade not found (local or system)");
            println!("      {}Install OpenCV with data files, or run setup again:{}", color::DIM, color::RESET);
            println!("        cargo run -- setup");
            had_errors = true;
        }
    }

    if Path::new(arcface::ARCFACE_MODEL).exists() {
        println!(
            "\n  {}[+]{} {}Verifying ArcFace model{}",
            color::CYAN, color::RESET, color::BOLD, color::RESET
        );
        match arcface::ArcFaceNet::load() {
            Ok(_) => ok("Model loads and initializes correctly"),
            Err(e) => {
                fail(&format!("Model failed to load: {e}"));
                had_errors = true;
            }
        }
    }

    println!();
    println!("  {}══════════════════════════════════════════════════════════════{}", color::CYAN, color::RESET);
    if had_errors {
        println!(
            "  {}{} Setup completed with errors.{} Fix the issues above and run again:",
            color::BOLD, color::YELLOW, color::RESET
        );
        println!("    cargo run -- setup");
    } else {
        println!(
            "  {}{} Setup complete!{} You're ready to go:",
            color::BOLD, color::GREEN, color::RESET
        );
        println!();
        println!("    {}cargo run -- enroll <name>{}    Enroll a face", color::BOLD, color::RESET);
        println!("    {}cargo run -- recognize{}        Live recognition", color::BOLD, color::RESET);
        println!("    {}cargo run -- hand{}             Hand tracking", color::BOLD, color::RESET);
    }
    println!("  {}══════════════════════════════════════════════════════════════{}", color::CYAN, color::RESET);
    println!();

    if had_errors {
        Err("Setup incomplete — see errors above".into())
    } else {
        Ok(())
    }
}
