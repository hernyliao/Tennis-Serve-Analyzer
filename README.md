# Tennis Serve Analyzer

An exploratory Python prototype that converts tennis serve videos into movement measurements and identifies player-specific associations with serve speed.

## Motivation

I wanted to explore whether a program could generate individualized tennis serve feedback from video. The project combines pose estimation, feature extraction, correlation analysis, and linear regression. Serve speed is the measured outcome; it does not capture other important aspects of serve quality, such as accuracy, spin, or consistency.

## How it works

1. Extract joint coordinates from serve videos using MediaPipe Pose.
2. Estimate a serve window and a contact-related frame using wrist movement heuristics.
3. Calculate frame-by-frame movement metrics, including knee angle, torso tilt, wrist position relative to the shoulder, and hip–shoulder separation.
4. Summarize each serve using extrema, measurements at the estimated contact frame, and relative timing features.
5. Group serves by player and compare features with recorded serve speeds.
6. Print exploratory suggestions based on correlation signs or regression coefficient signs.

The program does not track the ball or racket directly. Racket drop is represented by a wrist-to-shoulder position proxy, and contact is estimated from wrist movement rather than detected from ball impact.

## Files

| File | Purpose |
| --- | --- |
| `raw_data.py` | Reads `videos/*.mp4`, extracts pose coordinates, estimates the serve window, and writes `pose_data/*.csv`. |
| `save_metrics.py` | Converts pose coordinates into frame-by-frame metrics in `serve_metrics/*_metrics.csv`. |
| `save_features.py` | Summarizes each metrics file as one row and appends the rows to `serve_features.csv`. |
| `feedback.py` | Ranks individual features by absolute Pearson correlation with serve speed for each player. |
| `multidimensional_feedback.py` | Fits ordinary least-squares models with an intercept and ranks feature combinations by training R² for each player. |

## Inputs and naming

Place one serve video per MP4 file in a `videos` folder at the project root. Speed labels are read from filenames, not measured by the program.

Example: `playerA_95_serve01.mp4`

- `playerA` is the player identifier. Use letters only for compatibility with the current parser.
- `95` is an example speed label. Use the same speed unit throughout the dataset and state that unit when reporting results.
- `_serve01` distinguishes recordings.

The current filename parser requires an integer speed immediately before `_serve`. Separate recordings need unique filenames. The current pipeline assumes right-handed serves; left-handed support is incomplete.

## Environment and execution

The scripts import NumPy, OpenCV (`cv2`), and MediaPipe. They use the `mp.solutions.pose` API. Exact Python and package versions from the original working environment still need to be recorded before a reproducible environment can be specified.

Once a compatible environment and input videos are available, run these commands from the project root:

```bash
python raw_data.py
python save_metrics.py
python save_features.py
python feedback.py
python multidimensional_feedback.py
```

**Rerun behavior:** `save_features.py` appends all available serves each time it runs. To rebuild the feature table without duplicates, back up or remove the old `serve_features.csv` before running that step again.

No sample videos, feature table, or reproducible numerical outputs are included in the five source files supplied for this documentation. The complete pipeline has not been validated against a fresh dataset for this release.

## Analysis and interpretation

`feedback.py` calculates Pearson's r for each feature separately and reports the strongest absolute correlations. It requires at least three serves per player by default.

`multidimensional_feedback.py` uses NumPy least squares to fit models separately for each player. The current invocation considers every one- and two-feature combination and prints the five highest training R² values. It requires at least eight serves per player by default. Both scripts currently include `total_frames` among candidate predictors.

Training R² measures fit to the observations used to estimate the model. The current code does not implement a held-out evaluation or cross-validation. Searching many feature combinations on small samples can produce optimistic fits. Pearson's r and R² are distinct measures and should be labeled separately.

The presentation illustrates that observed relationships vary between players and that some recordings produce weak or inconsistent associations. Exact numerical results should be published with the corresponding dataset and reproducible analysis output.

Printed suggestions are hypotheses for further investigation. An association or coefficient sign does not demonstrate that changing a movement will increase serve speed.

## Limitations and implementation issues

- Small player-specific samples limit the strength and generalizability of conclusions.
- Pose-tracking errors, recording conditions, technique changes, and unreliable radar measurements affect input quality.
- Geometry uses MediaPipe image landmark coordinates, including estimated depth, rather than calibrated physical measurements. Camera perspective and coordinate scaling affect the interpretation of angles and distances.
- Contact detection and serve trimming use wrist-coordinate differences between detected frames, without dividing by elapsed time. Missing pose detections can affect timing features.
- `raw_data.py` checks for `pose_data/afterdata` before collecting existing outputs from `pose_data`. This can cause already processed videos to be reprocessed.
- The serve-start index can be negative, which can select an unintended slice. Very short pose sequences also need explicit handling.
- `knee_bend_deg` is the interior hip–knee–ankle angle: a straighter leg has a larger angle. The feature named `max_knee_bend` currently selects the maximum angle, not the deepest knee flexion. Knee timing and loading features depend on this selection and need review.
- Image y-coordinates increase downward. The comment describing the sign of the wrist-to-shoulder `racket_drop` measurement is reversed; the choice of its minimum as a drop feature also needs review.
- The output directions refer to numeric feature values. They should not be interpreted directly as validated coaching instructions.
- Empty inputs, malformed speed labels, constant features, and non-finite correlations need more robust handling.

## Next steps

Correct feature definitions and rerun handling; document the original environment and data-collection protocol; add a shareable example dataset and plots; then evaluate model stability on additional serves before assessing whether the suggestions improve performance.
