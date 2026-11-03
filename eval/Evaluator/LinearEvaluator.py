"""Pure translation evaluation adapted from Event-ST-PPP; see ../NOTICE.md."""
from .Evaluator import Evaluator


class LinearEvaluator(Evaluator):
    mode = 'trans'
    default_gt_format = 'velocity'
