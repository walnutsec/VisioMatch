# face_recon — real identity matching, not just detection

This replaces Gemini's "BIOMETRIC_TOPOLOGY" demo (which only *detects* faces and
always prints `STATUS: VERIFIED` regardless of who's in frame) with an actual
enroll → train → recognize pipeline using OpenCV's LBPH face recognizer.

## Setup

You need OpenCV **5.x** (or 4.x) built/packaged **with the contrib `face` module**. On Arch:

```bash
pacman -Ss opencv
```

Check whether `opencv::face` is available once you build — if the crate fails
to find the face module bindings, you'll need opencv built with
`-DOPENCV_EXTRA_MODULES_PATH` pointing at opencv_contrib, or grab an AUR
package that includes contrib.

```bash
cd face_recon
cargo build --release   # .cargo/config.toml sets OPENCV_PKGCONFIG_NAME=opencv5 automatically
```

If `Cargo.toml`'s pinned `opencv` version doesn't resolve against your system
OpenCV, just run:

```bash
cargo add opencv --features face
```

and let cargo pick a compatible version.

## Usage

```bash
# Enroll yourself (or anyone) — captures 30 samples, auto-trains after
cargo run -- enroll yourname

# Run live recognition
cargo run -- recognize

# Run hand tracking
cargo run -- hand
```

Enroll multiple people by running `enroll` again with a different name —
each run retrains the model on *all* enrolled identities.

## Tuning

`CONFIDENCE_THRESHOLD` in `main.rs` controls the match/unknown cutoff. LBPH
returns a **distance** (lower = more confident), not a percentage. To tune:
add a temporary `println!("{confidence}")` in `run_recognize`, watch the
values for a known face vs. a stranger, and set the threshold between the
two clusters you observe. Lighting and camera quality shift this a lot —
expect to tune it for your own setup.

## Known rough edge

OpenCV's `FaceRecognizer::write`/`read` are overloaded in C++ (file path vs.
`FileStorage`). Depending on your installed `opencv-rust` crate version, the
bindgen may have suffixed these as `write_1` / `read_1` instead of plain
`write` / `read`. If `recognizer.write(MODEL_PATH)` or `.read(MODEL_PATH)`
doesn't compile, that's almost certainly the fix — the compiler error will
usually suggest the correct name directly.

This was hand-written and reviewed against known opencv-rust API patterns
but not compiled in a live environment (no OpenCV/camera available here) —
treat the first `cargo build` as the real test.

## What's actually new vs. the original

- Real LBPH-based recognition instead of a hardcoded "VERIFIED" string
- Enrollment workflow with auto-retraining
- Fixed a crash bug: faces near frame edges or under 20px could send
  `Mat::roi` a negative width/height
- Stable per-face track IDs (simple centroid tracker) instead of a hash
  recomputed from scratch — and therefore different — every frame
- Color-coded MATCH/UNKNOWN status driven by an actual distance threshold
- OpenCV 5.0 support (auto-detects cascade path, handles renamed modules)
