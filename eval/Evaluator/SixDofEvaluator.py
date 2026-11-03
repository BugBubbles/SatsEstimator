"""Six-DoF evaluation adapted from Event-ST-PPP; see ../NOTICE.md."""
from .Evaluator import Evaluator


class SixDofEvaluator(Evaluator):
    mode = '6dof'
    dimensions = 6
    default_gt_format = 'velocity'
