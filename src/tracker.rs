//! Centroid-based face tracker with EMA-smoothed confidence.
//!
//! Associates detected faces across frames using centroid proximity
//! (proportional to face diagonal) and maintains stable track IDs.
//! Tracks are dropped after [`TRACK_MAX_MISSED`] consecutive frames
//! without an association.

/// Persistent per-face track used to stabilize the HUD across frames.
pub struct Track {
    pub id: u32,
    pub centroid: (f32, f32),
    pub name: String,
    pub confidence: f64,
    pub missed: u32,
}

/// A single face detection to be matched against existing tracks.
pub struct Detection {
    /// Centroid X coordinate (pixels).
    pub cx: f32,
    /// Centroid Y coordinate (pixels).
    pub cy: f32,
    /// Display name of the best-matching enrolled identity.
    pub name: String,
    /// Cosine similarity to the best match.
    pub confidence: f64,
    /// Diagonal length of the face bounding box (pixels), used to
    /// scale the association distance threshold.
    pub face_diag: f32,
}

/// Maximum centroid-distance multiplier relative to face diagonal for track association.
const TRACK_DIST_RATIO: f32 = 0.6;

/// Number of consecutive missed frames before a track is dropped.
const TRACK_MAX_MISSED: u32 = 20;

/// EMA weight for the new observation when smoothing track confidence.
const CONFIDENCE_EMA_NEW: f64 = 0.4;

/// Centroid-based multi-face tracker.

pub struct CentroidTracker {
    tracks: Vec<Track>,
    next_track_id: u32,
}

impl CentroidTracker {
    pub fn new() -> Self {
        Self {
            tracks: Vec::new(),
            next_track_id: 1,
        }
    }

    /// Update tracks with a new set of detections.
    ///
    /// Returns a vec of `(track_id, display_name, smoothed_confidence)` in
    /// the same order as the input detections.  Tracks not matched by any
    /// detection have their `missed` counter incremented; tracks exceeding
    /// [`TRACK_MAX_MISSED`] are dropped.
    pub fn update(&mut self, detections: &[Detection]) -> Vec<(u32, String, f64)> {
        let mut seen_track_ids = Vec::new();
        let mut results = Vec::new();

        for det in detections {
            let max_dist_sq = det.face_diag * TRACK_DIST_RATIO * det.face_diag * TRACK_DIST_RATIO;

            let track_idx = self.tracks.iter().position(|t| {
                let dx = t.centroid.0 - det.cx;
                let dy = t.centroid.1 - det.cy;
                dx * dx + dy * dy < max_dist_sq
            });

            let (track_id, result_name, result_conf) = match track_idx {
                Some(idx) => {
                    let t = &mut self.tracks[idx];
                    t.centroid = (det.cx, det.cy);
                    t.name = det.name.clone();
                    t.confidence = t.confidence * (1.0 - CONFIDENCE_EMA_NEW) + det.confidence * CONFIDENCE_EMA_NEW;
                    t.missed = 0;
                    (t.id, t.name.clone(), t.confidence)
                }
                None => {
                    let id = self.next_track_id;
                    // Handle id wrap-around
                    self.next_track_id = self.next_track_id.wrapping_add(1);
                    if self.next_track_id == 0 {
                        self.next_track_id = 1;
                    }

                    let name = det.name.clone();
                    let confidence = det.confidence;
                    self.tracks.push(Track {
                        id,
                        centroid: (det.cx, det.cy),
                        name: det.name.clone(),
                        confidence,
                        missed: 0,
                    });
                    (id, name, confidence)
                }
            };
            seen_track_ids.push(track_id);
            results.push((track_id, result_name, result_conf));
        }

        for t in self.tracks.iter_mut() {
            if !seen_track_ids.contains(&t.id) {
                t.missed += 1;
            }
        }
        self.tracks.retain(|t| t.missed < TRACK_MAX_MISSED);

        results
    }
}
