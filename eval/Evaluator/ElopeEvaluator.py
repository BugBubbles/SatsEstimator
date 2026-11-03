"""ELOPE evaluation from the local Event-ST-PPP extension; see ../NOTICE.md."""
from pathlib import Path
import numpy as np

from .Evaluator import check_times, load_table
from .SixDofEvaluator import SixDofEvaluator


class ElopeEvaluator(SixDofEvaluator):
    mode = 'elope'

    def get_gt(self):
        super().get_gt()
        filename = Path(self.depth_file) if self.depth_file else self.dataset.with_name('depth.txt')
        self.depth = load_table(filename, (2,))
        self.depth[:, 0] += self.applied_gt_time_offset
        check_times(self.depth, 'depth')
        if np.any(self.depth[:, 1] <= 0):
            raise ValueError('Depth/range measurements must be positive')

    @staticmethod
    def score(gt, es, depth):
        return float(np.mean(np.linalg.norm(gt - es, axis=1) / depth))
