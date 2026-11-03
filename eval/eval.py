"""Evaluate existing result logs from the command line, without GUI dependencies."""
import argparse
import json
from pathlib import Path
import sys

if __package__:
    from .Evaluator import AngularEvaluator, LinearEvaluator, SixDofEvaluator, ElopeEvaluator
else:
    from Evaluator import AngularEvaluator, LinearEvaluator, SixDofEvaluator, ElopeEvaluator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--transformation', choices=('rot', 'trans', '6dof', 'elope'), required=True)
    parser.add_argument('-g', '--gt-file', '--gt_file', dest='gt_file', type=Path, required=True,
                        help='IMU text for rot; velocity groundtruth.txt for other modes')
    parser.add_argument('-e', '--estimate', type=Path, required=True, help='ResultLogger TXT file')
    parser.add_argument('--gt-format', choices=('imu', 'velocity'), help='Default: imu for rot, velocity otherwise')
    parser.add_argument('--gt-time-offset', type=float,
                        help='Seconds added to GT/depth timestamps; default -0.0024 for IMU, 0 otherwise')
    parser.add_argument('--depth-file', type=Path, help='ELOPE range TXT; default depth.txt beside ground truth')
    parser.add_argument('-o', '--output', type=Path, help='Save metrics as JSON')
    parser.add_argument('--plot', type=Path, help='Save comparison plot using matplotlib (no display needed)')
    args = parser.parse_args(argv)
    evaluators = {'rot': AngularEvaluator, 'trans': LinearEvaluator, '6dof': SixDofEvaluator, 'elope': ElopeEvaluator}
    if args.depth_file is not None and args.transformation != 'elope':
        parser.error('--depth-file applies only to elope')
    try:
        evaluator = evaluators[args.transformation](args.gt_file, args.estimate,
            gt_format=args.gt_format, gt_time_offset=args.gt_time_offset, depth_file=args.depth_file)
        result = evaluator()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        if args.plot:
            evaluator.plot(args.plot)
    except (OSError, ValueError, ImportError) as error:
        print(f'Evaluation failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
