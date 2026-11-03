"""Pure rotation evaluation adapted from Event-ST-PPP; see ../NOTICE.md."""
from .Evaluator import Evaluator


class AngularEvaluator(Evaluator):
    mode = 'rot'
    default_gt_format = 'imu'
