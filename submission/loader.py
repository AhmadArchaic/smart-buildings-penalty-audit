import os, sys, glob, traceback
from pathlib import Path
import numpy as np
D = next(str(p.parent) for p in [*Path("/kaggle/input").glob("*/kaggle_evaluation"), *Path("/kaggle/input").glob("*/*/kaggle_evaluation")] if "smart-buildings" in str(p)) + "/"
sys.path[:0] = [D + "kaggle_evaluation", D]
P = sorted(glob.glob("/kaggle/input/**/policies/ctrl_x.py", recursive=True))[0]
sys.path.insert(0, os.path.dirname(P))
print("data:", D, "policy:", P, flush=True)
from ctrl_x import CtrlX
# Our best leaderboard entry (0.00983) uses CtrlX with its default parameters (no tank hold).
# Round 1 tank-hold variant: {"water_idle_hold": True, "water_peak_hold_hours": 30, "water_rise_guard_K": 2.5, "water_offpeak_floor": 0}
CtrlX.PARAMS = {**CtrlX.PARAMS, **{}}
from kaggle_evaluation import smart_buildings_inference_server as sbis
def _fix(a, s, t):
    a = np.nan_to_num(np.asarray(a, dtype=np.float64).reshape(s.shape), nan=0.0, posinf=0.0, neginf=0.0)
    if t is np.int32: a = np.rint(a)
    return np.clip(a, s.low, s.high).astype(t)
class SubmissionPolicy(sbis.Policy):
    def __init__(self, md, spec):
        super().__init__(md, spec)
        self.c, self.d = spec["continuous_action"], spec["discrete_action"]
        self.safe = {"continuous_action": _fix(np.zeros(self.c.shape), self.c, np.float32), "discrete_action": _fix(np.ones(self.d.shape), self.d, np.int32)}
        self.n = 0
        self.impl = CtrlX(md, spec)
    def get_action(self, obs):
        try:
            a = self.impl.get_action(obs)
            return {"continuous_action": _fix(a["continuous_action"], self.c, np.float32), "discrete_action": _fix(a["discrete_action"], self.d, np.int32)}
        except Exception:
            self.n += 1
            if self.n == 1: traceback.print_exc()
            return {k: v.copy() for k, v in self.safe.items()}
sbis.SmartBuildingsInferenceServer(SubmissionPolicy).run(competition_data_folder=D)
