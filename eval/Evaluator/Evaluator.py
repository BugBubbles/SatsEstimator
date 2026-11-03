"""Event-ST-PPP evaluation methods adapted for SATS. See ../NOTICE.md."""

from pathlib import Path
import warnings

import numpy as np
from scipy.interpolate import interp1d


def load_table(filename, columns=None):
    """Read whitespace-delimited numeric text, including # and % comments."""
    with Path(filename).open() as stream:
        rows = (line for line in stream if line.strip() and not line.lstrip().startswith(('#', '%')))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            table = np.loadtxt(rows, ndmin=2)
    if not table.size:
        raise ValueError(f'{filename}: no numeric rows')
    if columns is not None and table.shape[1] not in columns:
        raise ValueError(f'{filename}: expected {columns} columns, got {table.shape[1]}')
    if not np.isfinite(table).all():
        raise ValueError(f'{filename}: contains NaN or infinity; use ground truth with complete labels')
    return table


def check_times(table, label):
    if len(table) < 2 or np.any(np.diff(table[:, 0]) <= 0):
        raise ValueError(f'{label}: need at least two rows with strictly increasing timestamps')


def ratio(value, denominator):
    # Undefined normalization is represented by null in JSON, rather than Inf/NaN.
    return float(value / denominator) if denominator > 0 else None


class Evaluator:
    """Shared interpolation and original component-wise error calculations."""

    mode = 'rot'
    dimensions = 3
    default_gt_format = 'imu'

    def __init__(self, dataset, res_dir, *, gt_format=None, gt_time_offset=None, depth_file=None):
        self.dataset = Path(dataset)
        self.gt_format = gt_format or self.default_gt_format
        self.gt_time_offset = gt_time_offset
        self.depth_file = depth_file
        self.get_gt()
        self.get_es(res_dir)
        self.res = {}

    def get_gt(self):
        if self.gt_format == 'imu':
            if self.mode != 'rot':
                raise ValueError('IMU ground truth is supported only for rotation evaluation')
            raw = load_table(self.dataset, (7,))
            self.gt = raw[:, [0, 4, 5, 6]].copy()
            offset = -0.0024 if self.gt_time_offset is None else self.gt_time_offset
        elif self.gt_format == 'velocity':
            raw = load_table(self.dataset, (4, 7) if self.dimensions == 3 else (7,))
            if self.mode == 'rot' and raw.shape[1] == 7:
                self.gt = raw[:, [0, 4, 5, 6]].copy()
            else:
                self.gt = raw[:, :self.dimensions + 1].copy()
            offset = 0.0 if self.gt_time_offset is None else self.gt_time_offset
        else:
            raise ValueError(f'Unknown ground truth format: {self.gt_format}')
        if not np.isfinite(offset):
            raise ValueError('Ground truth time offset must be finite')
        self.gt[:, 0] += offset
        self.applied_gt_time_offset = float(offset)
        check_times(self.gt, 'ground truth')
        self.gt_max = np.max(self.gt[:, 1:])
        self.gt_min = np.min(self.gt[:, 1:])

    def get_es(self, filename):
        self.es = load_table(filename, (self.dimensions + 3, self.dimensions + 4))
        if np.any(self.es[:, 2] < self.es[:, 1]):
            raise ValueError('Estimate window end precedes its start')

    def data_selection(self, data1, data2, intp=True):
        if not intp:
            raise ValueError('Only the original linear interpolation evaluation is supported')
        values = interp1d(data1[:, 0], data1[:, 1:], axis=0, kind='linear', bounds_error=True)(data2[:, 0])
        return np.column_stack((data2[:, 0], values))

    def simple_pre(self, save=False):
        timestamps = (self.es[:, 1] + self.es[:, 2]) / 2
        lower, upper = self.gt[0, 0], self.gt[-1, 0]
        if hasattr(self, 'depth'):
            lower = max(lower, self.depth[0, 0])
            upper = min(upper, self.depth[-1, 0])
        self.valid = (timestamps >= lower) & (timestamps <= upper)
        if not self.valid.any():
            raise ValueError('Estimate midpoints have no overlap with ground truth/depth timestamps')
        estimates = np.column_stack((timestamps[self.valid], self.es[self.valid, -self.dimensions:]))
        self.aligned_es = estimates
        self.aligned_gt = self.data_selection(self.gt, estimates)
        return self.aligned_gt, estimates

    @staticmethod
    def rms(data1, data2):
        return np.sqrt(np.mean(np.power(data1 - data2, 2)))

    def metrics(self, gt, es, *, angular=False, six_dof=False):
        expected, predicted = gt[:, 1:], es[:, 1:]
        error = (expected - predicted).reshape(-1)
        rms = self.rms(expected.reshape(-1), predicted.reshape(-1))
        # Preserve the local source's inclusion of time in the 6-DOF RMS range.
        rms_range = np.ptp(gt) if six_dof else self.gt_max - self.gt_min
        value_range = np.ptp(expected)
        scale = 180.0 / np.pi if angular else 1.0
        rms_relative = ratio(rms, rms_range)
        max_relative = ratio(np.max(np.abs(error)), value_range)
        if angular and six_dof:
            if rms_relative is not None:
                rms_relative = rms_relative / np.pi * 180.0
            if max_relative is not None:
                max_relative = max_relative / np.pi * 180.0
        return {
            'unit': 'deg/s' if angular else 'm/s',
            'rms': float(rms / np.pi * 180.0 if angular else rms),
            'rms_r%': rms_relative,
            'max_err': float(np.max(np.abs(error)) * scale),
            'max_err_r%': max_relative,
            'mean_err': float(np.mean(np.abs(error)) * scale),
            'mean_err_r%': ratio(np.mean(np.abs(error)), value_range),
            'mean_err_x': float(np.mean(np.abs(expected[:, 0] - predicted[:, 0])) * scale),
            'mean_err_y': float(np.mean(np.abs(expected[:, 1] - predicted[:, 1])) * scale),
            'mean_err_z': float(np.mean(np.abs(expected[:, 2] - predicted[:, 2])) * scale),
            'std': float(np.std(error) * scale),
            'std_r%': ratio(np.std(error), value_range),
        }

    def simple_evaluate(self):
        gt, es = self.simple_pre()
        if self.dimensions == 3:
            key = 'rotation' if self.mode == 'rot' else 'translation'
            self.res = {key: self.metrics(gt, es, angular=self.mode == 'rot')}
        else:
            indices_v, indices_w = [0, 1, 2, 3], [0, 4, 5, 6]
            self.res = {
                'translation': self.metrics(gt[:, indices_v], es[:, indices_v], six_dof=True),
                'rotation': self.metrics(gt[:, indices_w], es[:, indices_w], angular=True, six_dof=True),
            }
        if hasattr(self, 'depth'):
            depths = self.data_selection(self.depth, es)[:, 1]
            self.res['score'] = self.score(gt[:, 1:4], es[:, 1:4], depths)
        self.res.update({
            'mode': self.mode,
            'metric_convention': 'Event-ST-PPP legacy',
            'samples': int(self.valid.sum()),
            'excluded_outside_time_support': int((~self.valid).sum()),
            'gt_time_offset_s': self.applied_gt_time_offset,
            'runtime_mean_s': float(np.mean(self.es[self.valid, 3])) if self.es.shape[1] == self.dimensions + 4 else None,
        })
        return self.res

    def __call__(self, save_res=False):
        if save_res:
            raise ValueError('Use the command-line --output option to save JSON without a file dialog')
        result = self.simple_evaluate()
        self.print_result()
        return result

    def print_result(self):
        print(f'Mode: {self.mode}; samples: {self.res["samples"]}; excluded outside time support: {self.res["excluded_outside_time_support"]}')
        print('Metric convention: Event-ST-PPP legacy (see eval/README.md)')
        for group in ('translation', 'rotation'):
            if group in self.res:
                print(f'\n{group} [{self.res[group]["unit"]}]')
                for name, value in self.res[group].items():
                    if name != 'unit':
                        print(f'  {name:16s} {value:.10g}' if value is not None else f'  {name:16s} undefined')
        if 'score' in self.res:
            print(f'\nELOPE score [1/s]: {self.res["score"]:.10g}')
        if self.res['runtime_mean_s'] is not None:
            print(f'Mean runtime [s]: {self.res["runtime_mean_s"]:.10g}')

    def plot(self, filename):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(self.dimensions, 1, figsize=(11, 2.5 * self.dimensions), squeeze=False)
        for index, axis in enumerate(axes[:, 0]):
            angular = self.mode == 'rot' or (self.dimensions == 6 and index >= 3)
            scale = 180.0 / np.pi if angular else 1.0
            axis.plot(self.aligned_gt[:, 0], self.aligned_gt[:, index + 1] * scale, label='ground truth')
            axis.plot(self.aligned_es[:, 0], self.aligned_es[:, index + 1] * scale, '--', label='estimate')
            axis.set_ylabel(f'{"w" if angular else "v"}{"xyz"[index % 3]} [{"deg/s" if angular else "m/s"}]')
            axis.grid(True)
        axes[0, 0].legend()
        axes[-1, 0].set_xlabel('time [s]')
        fig.tight_layout()
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename)
        plt.close(fig)
