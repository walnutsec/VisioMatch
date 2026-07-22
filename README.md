<div align="center">

# VisioMatch

**Real-time face recognition and hand tracking powered by ArcFace deep learning, built in Rust with OpenCV 5.x.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x-green.svg)](https://opencv.org/)

</div>

---

## Overview

VisioMatch is a Rust-based computer vision system that performs **real-time face recognition** using ArcFace (MobileFaceNet) deep learning embeddings. It identifies enrolled individuals via cosine similarity matching — no retraining required. A secondary hand-tracking mode uses MediaPipe landmarks for 21-keypoint hand skeleton visualization.

### Pipeline

```
Camera Frame
  ├── Face Mode: Haar Cascade (½ resolution) → ArcFace MobileFaceNet (112×112)
  │     → 512-d Embedding → Cosine Similarity → MATCH / UNKNOWN
  └── Hand Mode: YCrCb Skin Segmentation → MediaPipe ONNX (224×224)
        → 21 Landmarks → Skeleton + Handedness Classification
```

### Key Features

- **ArcFace MobileFaceNet** (`w600k_mbf`) — 99.4% accuracy on LFW, 512-d embeddings
- **One-shot enrollment** — 5 face samples, averaged and stored as a single vector
- **No retraining** — add identities by storing embedding vectors in JSON
- **Stable per-face tracking** — centroid tracker with EMA-smoothed similarity scores
- **HUD rendering** — cyberpunk-style corner brackets, structural mesh, identity overlay
- **Hand tracking** — MediaPipe 21-keypoint skeleton with handedness classification
- **OpenCL acceleration** — offloads DNN inference to GPU when available (falls back to CPU)
- **OpenCV 5.x native** — handles renamed modules (`features2d` → `features`, `calib3d` → `calib`)

---

## Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| **Rust** | ≥ 1.70 | Stable toolchain |
| **OpenCV** | 5.x (or 4.x) | With development headers and `pkg-config` support |
| **Clang** | Any recent | Required by `opencv` crate's `bindgen` pass |

### System Packages

**Arch Linux:**
```bash
sudo pacman -S opencv vtk hdf5 clang pkg-config
```

**Ubuntu / Debian:**
```bash
sudo apt install libopencv-dev clang pkg-config
```

**Fedora:**
```bash
sudo dnf install opencv-devel clang pkg-config
```

> [!NOTE]
> The project ships with `.cargo/config.toml` that sets `OPENCV_PKGCONFIG_NAME=opencv5`.
> If you are using OpenCV 4.x, change this to `opencv4` or remove the line entirely.

---

## Installation

```bash
git clone https://github.com/walnutsec/VisioMatch.git
cd VisioMatch
cargo build --release
```

### Quick Setup (Recommended)

Run the interactive setup wizard — it downloads models, creates directories, and verifies your environment automatically:

```bash
cargo run --release -- setup
```

The wizard will:
1. Check for `curl` or `wget`
2. Create required directories (`data/models/`, `data/faces/`)
3. Download the ArcFace model (~13 MB, required)
4. Optionally download the hand landmark model (~5 MB)
5. Verify the Haar cascade and model loading

<details>
<summary><b>Manual setup (alternative)</b></summary>

```bash
# Required: ArcFace model
mkdir -p data/models
wget -O data/models/w600k_mbf.onnx \
  https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx

# Optional: Hand landmark model
wget -O data/models/handpose_estimation_mediapipe_2023feb.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx
```

</details>

---

## Usage

### Enroll a Face

Capture 5 face samples and store the averaged embedding:

```bash
cargo run --release -- enroll <name>
```

- Position your face in the camera frame (exactly one face must be visible)
- The system captures a sample every 500ms automatically
- Move your head slightly between captures for robustness
- Press `q` to cancel early

Enroll multiple people by running `enroll` again with different names. Each identity is stored as a 512-float vector in `data/embeddings.json`.

### Live Recognition

```bash
cargo run --release -- recognize
```

- Faces are detected, identified, and annotated in real time
- The HUD displays track ID, identity name, similarity score, and match status
- Press `q` or `Esc` to exit

### Hand Tracking

```bash
cargo run --release -- hand
```

- Detects up to 2 hands with left/right classification
- Renders a 21-keypoint skeleton overlay with confidence scores
- Press `q` or `Esc` to exit

---

## Project Structure

```
VisioMatch/
├── .cargo/
│   └── config.toml       # pkg-config setup for OpenCV 5.x
├── data/
│   ├── models/            # ONNX model files (not tracked in git)
│   ├── embeddings.json    # Enrolled identity database
│   └── faces/             # Reference face crops from enrollment
├── src/
│   ├── main.rs            # CLI entry point and module declarations
│   ├── arcface.rs         # ArcFace model wrapper, embedding math, JSON persistence
│   ├── camera.rs          # Camera setup (720p @ 30fps)
│   ├── detect.rs          # Haar cascade face detection (half-resolution)
│   ├── enroll.rs          # Enrollment workflow: capture → embed → save
│   ├── recognize.rs       # Live recognition with track system + HUD
│   ├── tracker.rs         # Centroid-based face tracker with EMA smoothing
│   ├── hands.rs           # Hand tracking (MediaPipe landmark model)
│   └── setup.rs           # Interactive first-time setup wizard
├── Cargo.toml
├── LICENSE                # MIT
└── README.md
```

---

## Configuration & Tuning

### Cosine Similarity Threshold

The match/unknown decision is controlled by `COSINE_THRESHOLD` in [`src/arcface.rs`](src/arcface.rs):

| Range | Typical meaning |
|-------|-----------------|
| **0.5 – 0.8** | Same person (high confidence) |
| **0.3 – 0.5** | Ambiguous — may need more enrollment samples |
| **< 0.3** | Different person |

The default is **0.4**. Adjust based on your deployment:
- **Raise** for stricter matching (fewer false positives, more false rejections)
- **Lower** to reduce false rejections at the cost of more false matches

> [!TIP]
> Watch the `SIM` value in the recognition HUD while a known face is in frame to calibrate the threshold for your camera and lighting conditions.

### OpenCL Acceleration

DNN inference uses `DNN_BACKEND_OPENCV` + `DNN_TARGET_OPENCL` by default. This offloads matrix operations to the GPU via OpenCL when a compatible device is available (e.g., Intel UHD iGPU). If no OpenCL runtime is installed, it falls back to CPU transparently.

To install OpenCL support on Intel:
```bash
# Arch Linux
sudo pacman -S intel-compute-runtime

# Ubuntu / Debian
sudo apt install intel-opencl-icd
```

---

## Troubleshooting

### Build Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `opencv5.pc not found` | OpenCV 4.x installed, not 5.x | Change `OPENCV_PKGCONFIG_NAME` in `.cargo/config.toml` to `opencv4` |
| `bindgen` crash / OOM | `default-features = true` on the `opencv` crate | Ensure `default-features = false` in `Cargo.toml` (already set) |
| `calib3d` / `features2d` linker errors | `opencv` crate < 0.99 with OpenCV 5.x | Use `opencv = "0.99"` or newer |

### Runtime Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ArcFace model not found` | Missing ONNX model file | Run the `wget` command from the [Installation](#installation) section |
| `Could not open camera device` | No camera or permission denied | Check `ls /dev/video*` and ensure your user is in the `video` group |
| `Haar cascade not found` | OpenCV installed without data files | Install the full OpenCV package (not just the library) |

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. **Fork** the repository and create a feature branch
2. **Follow** existing code style (Rust 2021 edition, `cargo clippy` clean)
3. **Test** your changes with `cargo check && cargo clippy`
4. **Document** public functions with `///` doc comments
5. **Submit** a pull request with a clear description of changes

### Development Setup

```bash
# Check compilation
cargo check

# Run lints
cargo clippy

# Build optimized binary
cargo build --release
```

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright © 2026 walnutsec
