# Hybrid KLT Tracker (`HybridKLTAutomation`)

A mission-tunable MATLAB **Video Labeler Automation Algorithm** combining **Gaussian Mixture Model (GMM) Foreground Blob Detection**, a **Linear Kalman Filter**, and **Kanade-Lucas-Tomasi (KLT) Optical Flow Feature Tracking** for semi-automated bounding-box annotation across UAV, aerial surveillance, and ground tracking datasets.

**Author:** Zarif Bin Akhtar

---

## Video Demonstration

See [`demo/video_demo.mp4`](demo/video_demo.mp4) for a walkthrough of the Hybrid KLT Tracker running inside MATLAB Video Labeler.

---

## Architecture Overview

1. **Kalman Prediction (`Part 1`):** Uses a constant-velocity 4-state model $[x, y, v_x, v_y]^T$ to predict target position across frames and bridge temporary occlusions.
2. **Foreground Blob Detection (`Part 2`):** Applies `vision.ForegroundDetector` followed by configurable morphological opening (`filterSize`), closing (`second_filterSize`), and hole filling, associating the closest valid blob within `MaxAssociationDist`.
3. **KLT Feature Tracking (`Part 2`):** Maintains local corner features (`detectMinEigenFeatures` + `vision.PointTracker`) inside the target ROI and automatically re-seeds features when tracked points drop below threshold.
4. **Kalman State Update (`Part 3`):** Fuses valid blob detections with predicted kinematics using tunable process noise ($Q_k$) and measurement noise ($R_k$).

---

## Requirements

- **MATLAB** R2020b or newer recommended
- **Computer Vision Toolbox** (`vision.labeler.AutomationAlgorithm`, `vision.ForegroundDetector`, `vision.BlobAnalysis`, `vision.PointTracker`, `detectMinEigenFeatures`)
- **Image Processing Toolbox** (`imopen`, `imclose`, `imfill`, `strel`)

---

## Installation & Video Labeler Setup

1. Clone or download this repository:
   ```bash
   git clone https://github.com/Akhtar-Zarif/Hybrid-KLT-Tracker.git
   ```
2. Open MATLAB and add the repository folder to your path:
   ```matlab
   addpath(genpath('path/to/Hybrid-KLT-Tracker'));
   savepath;
   ```
3. Launch the **Video Labeler** app in MATLAB:
   ```matlab
   videoLabeler
   ```
4. Load your video or image sequence and create a **Rectangle** ROI label definition.
5. Draw an initial bounding box around the target in the first frame.
6. In the **Automate** tab, select **Hybrid KLT Tracker** from the algorithm dropdown.
7. Click **Settings** to pick a **Mission Preset** or fine-tune individual parameters, then click **Automate** -> **Run**.

---

## Mission Presets & Tuning Guide

Because target scale, camera altitude, frame rate, and background clutter vary widely across missions, `HybridKLTAutomation` includes built-in presets in the **Settings** dialog as well as full manual overrides:

| Parameter | Default | High-Altitude / Small UAV | Fast / Low-Altitude Drone | Ground Vehicle / Pedestrian | Cluttered Background | Effect of Increasing |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `minBlobArea` | `200` | `40` | `150` | `400` | `250` | Ignores smaller noise blobs; may miss distant targets |
| `filterSize` (Open) | `3` | `2` | `3` | `5` | `5` | Removes speckle noise; erodes thin target edges |
| `second_filterSize` (Close) | `25` | `15` | `25` | `35` | `30` | Merges fragmented target parts into one bounding box |
| `MaxAssociationDist` | `200` | `120` | `300` | `180` | `150` | Allows larger frame-to-frame jumps; increases ID-switch risk |
| `NumPyramidLevels` | `3` | `4` | `4` | `3` | `3` | Handles larger inter-frame pixel displacements |
| `MaxBidirectionalError` | `0.2` | `0.5` | `1.0` | `0.2` | `0.2` | Retains points under motion blur; lowers point precision |
| `BlockSize` | `[31 31]` | `[21 21]` | `[41 41]` | `[31 31]` | `[31 31]` | Integrates larger local texture patches |
| `MinFeatureQuality` | `0.01` | `0.005` | `0.01` | `0.01` | `0.02` | Restricts KLT tracking to sharper, stronger corners |
| `Qk` (Process Noise) | `1e-2` | `5e-2` | `1e-1` | `1e-2` | `1e-2` | Reacts faster to sharp turns and sudden acceleration |
| `Rk` (Meas. Noise) | `5e-2` | `2e-2` | `5e-2` | `5e-2` | `1e-1` | Smooths noisy detector centroids; relies more on prediction |

### Quick Tuning Tips
- **Target disappears during turns or high-G maneuvers:** Increase `Qk` (`5e-2` to `1e-1`) and `MaxAssociationDist` (`250–350`).
- **Distant / tiny UAV is filtered out:** Lower `minBlobArea` (`20–60`) and `filterSize` (`2`).
- **Wind, foliage, or moving shadows create false detections:** Increase `minBlobArea`, `filterSize`, and `Rk`.
- **Single target splits into multiple boxes:** Increase `second_filterSize` (`30–45`).

---

## Repository Structure

```text
Hybrid-KLT-Tracker/
├── HybridKLTAutomation.m       # Main MATLAB Video Labeler AutomationAlgorithm class
├── presets/
│   └── loadMissionConfig.m     # Helper utility to inspect/export preset configurations
├── demo/
│   ├── README.md               # Instructions for adding and linking video_demo.mp4
│   └── video_demo.mp4          # Demonstration video of Hybrid KLT Tracker in action
├── .gitignore                  # Ignores MATLAB autosaves, temp files, and local datasets
├── LICENSE                     # MIT License
└── README.md                   # Documentation, setup instructions, and tuning guide
```

---

## License

Released under the [MIT License](LICENSE).
