# Velocity evaluation

These scripts port the rotation, translation, six-DoF, and ELOPE evaluators from the local Event-ST-PPP project. Run them directly from the repository root; paths are explicit and dataset names are not inferred from directory names. Only NumPy and SciPy are needed for metrics. Matplotlib is optional for `--plot`. No installation is performed by the scripts.

## Commands

```bash
# ECD / DAVIS 240C: compare angular velocities with IMU gyroscope measurements
python eval/eval.py -t rot \
  -g /path/to/datasets/Davis_240C/boxes_rotation/imu.txt \
  -e /path/to/results/rotation.txt

# MVSEC: pure translation
python eval/eval.py -t trans \
  -g /path/to/datasets/MVSEC/indoor_flying1/groundtruth.txt \
  -e /path/to/results/translation.txt

# MVSEC: six-DoF
python eval/eval.py -t 6dof \
  -g /path/to/datasets/MVSEC/indoor_flying1/groundtruth.txt \
  -e /path/to/results/6dof.txt

# ELOPE: six-DoF metrics and depth-normalized translation score
python eval/eval.py -t elope \
  -g /path/to/datasets/ELOPE/0000/groundtruth.txt \
  -e /path/to/results/elope.txt
```

For ELOPE, `depth.txt` is loaded beside `groundtruth.txt`. Override this with `--depth-file /path/to/depth.txt` if needed. The outputs from `scripts/mvsec_gt.py` and `scripts/elope_gt.py` can be used directly.

Each command prints metrics. Add `--output ./results/metrics.json` to save JSON, and `--plot ./results/velocities.png` to save a comparison plot without a display. The entry point also supports `python -m eval.eval ...`; the old `--gt_file` spelling remains accepted. Use `python eval/eval.py --help` for all arguments.

## Input formats

All inputs are whitespace-delimited TXT files; blank lines and `#` or `%` comments are skipped. Velocities in inputs use m/s and rad/s; all timestamps use seconds with matching origins.

| Input | Columns |
|-------|---------|
| Rotation estimate | `index t_start t_end runtime wx wy wz` |
| Translation estimate | `index t_start t_end runtime vx vy vz` |
| Six-DoF / ELOPE estimate | `index t_start t_end runtime vx vy vz wx wy wz` |
| ECD rotation reference (`imu.txt`) | `t ax ay az wx wy wz` |
| Translation reference | `t vx vy vz` or `t vx vy vz wx wy wz` |
| Six-DoF / ELOPE reference | `t vx vy vz wx wy wz` |
| ELOPE range (`depth.txt`) | `t range_m` |

The estimate files are the existing C++ `ResultLogger` outputs; no reformatting is needed. Legacy estimate files with the `runtime` column omitted are also accepted, in which case runtime is reported as unavailable. A legacy loss column in the runtime position does not affect accuracy metrics, but its reported mean is not a timing measurement.

Rotation defaults to the ECD IMU format and adds **−0.0024 s** to its timestamps, matching Event-ST-PPP. ECD `groundtruth.txt` contains poses and must not be passed as angular velocity ground truth. For a rotation reference already stored as `t wx wy wz` (or the seven-column velocity format), use `--gt-format velocity`; its default time offset is zero. `--gt-time-offset` overrides either default. For ELOPE, an explicit offset is applied to both velocity and range timestamps.

Reference timestamps must be strictly increasing, with at least two rows. Input NaNs/infinities, malformed column counts, reversed estimate windows, nonpositive ranges, and files without overlapping times produce an error. ELOPE evaluation needs complete ground truth, such as the training split.

## Metric convention

The default convention deliberately reproduces the local source scripts, including their historical normalization behavior, for comparison with existing experiments:

- The estimate time is `(t_start + t_end) / 2`. Reference velocities and ELOPE range are linearly interpolated at that time. No extrapolation is performed. Windows outside the reference time support are excluded and their count is reported; for ELOPE, both reference streams must cover a window midpoint. This is the only automatic time selection; full reference sequences still define the original rotation/translation RMS normalization range.
- RMS, maximum absolute error, mean absolute error, and population standard deviation are computed over the flattened three-component errors. Per-axis mean absolute errors are also reported. Six-DoF evaluates translation and rotation separately.
- Translation errors use m/s. Angular absolute errors use deg/s; input angular velocities remain rad/s.
- For rotation/translation alone, normalized RMS divides by the full reference velocity range, excluding timestamps. For six-DoF/ELOPE, the source takes the range of the **interpolated `[t, vx, vy, vz]` or `[t, wx, wy, wz]` table, including time**. This behavior is preserved.
- The source additionally converts the six-DoF/ELOPE **rotation normalized RMS and normalized maximum error** with `value / pi * 180`. This behavior is preserved. The normalized maximum error uses the interpolated velocity range before that conversion. Normalized mean error and standard deviation use the interpolated velocity range without time or an angular conversion.
- Names ending in `%` retain the source's labels: their values are **not multiplied by 100**. They should not be interpreted as conventional percentages, especially the historical six-DoF rotation RMS entry.
- A zero normalization range produces `undefined` in the console and `null` in JSON. Absolute metrics remain available. Runtime, when present, is the mean of the retained rows' runtime column, in seconds.

ELOPE additionally reports the unchanged score:

```text
score = mean(norm(v_groundtruth - v_estimate, axis=1) / interpolated_range)
```

This is a mean over retained estimate windows, in 1/s; no resampling to an external competition submission grid is added.

## Source and license

See [NOTICE.md](NOTICE.md) for attribution and adaptation details. The adapted evaluation code retains the source project's [CC BY-NC-SA 4.0 license](LICENSE.md).
