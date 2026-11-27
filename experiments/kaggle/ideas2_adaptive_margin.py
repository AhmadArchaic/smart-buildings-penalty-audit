import os, sys, glob, csv, itertools
from pathlib import Path
from multiprocessing import Pool
D = os.environ.get("SB_D") or next(str(p.parent) for p in [*Path("/kaggle/input").glob("*/smart_buildings"), *Path("/kaggle/input").glob("*/*/smart_buildings")] if "smart-buildings" in str(p))
POL = os.environ.get("SB_POL") or os.path.dirname(sorted(glob.glob("/kaggle/input/**/policies/ctrl_x.py", recursive=True))[0])
OUT = os.environ.get("SB_OUT", "/kaggle/working/ideas2_r2.csv")
STEPS = int(os.environ.get("SB_STEPS", 2016))
V = {"base": ({}, None), "adapt_255": ({}, 255.0), "adapt_262": ({}, 262.0),
     "adapt_255_floor325": ({"water_floor": 325.0}, 255.0), "lead_0.25h": ({"arrival_lead_hours": 0.25}, None)}

def run(job):
    name, scen = job
    sys.path[:0] = [D, POL]; os.chdir(D)
    import numpy as np
    from smart_buildings import evaluate as ev
    from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
    from ctrl_x import CtrlX
    params, cold = V[name]
    class P(CtrlX):
        PARAMS = {**CtrlX.PARAMS, **params}
        def get_action(self, obs):
            if cold is not None and self.outside_obs:
                t = self._temperature(obs, self.outside_obs[0], "outside_air_temperature_sensor", 283.0)
                h = getattr(self, "_oat", [])
                h.append(t); self._oat = h[-288:]
                self.parameters["comfort_margin"] = 0.04 if (len(self._oat) >= 36 and sum(self._oat) / len(self._oat) < cold) else 0.08
            return super().get_action(obs)
    d = Path("smart_buildings/training_scenarios") / scen
    env = ev.load_hybrid_environment(d / "config.gin")
    tr = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    env.metrics_path = None; env._occupancy_normalization_constant = 125.0
    md = ev.create_building_meta_data(env, d)
    ev.run_policy(env, tr, ev.initialize_policy(P, md, env.action_spec()), STEPS)
    r = np.array(tr.rewards, float); c = tr.reward_components or []
    m = lambda k: float(np.mean([x.get(k, 0) for x in c])) if c else None
    row = [name, scen, len(r), float(-r.mean()), m("normalized_productivity_regret"), m("normalized_energy_cost"), m("normalized_carbon_emission")]
    print(row, flush=True); return row

def safe(j):
    try: return run(j)
    except Exception as e: print("FAIL", j, repr(e), flush=True); return [j[0], j[1], "nan", "nan", "nan", "nan", "nan"]

if __name__ == "__main__":
    four = ["building_train_1/" + w for w in ("phoenix_5", "denver_7", "fairbanks_1", "singapore_8")]
    jobs = [(v, s) for v in V for s in four]
    print(len(jobs), "jobs; round2:", os.path.exists(os.path.join(D, "smart_buildings", "check_return_air.py")), flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["variant", "scenario", "steps", "score", "comfort", "energy", "carbon"])
        with Pool(os.cpu_count()) as pool:
            for row in pool.imap_unordered(safe, jobs): w.writerow(row); f.flush()
    print("DONE", flush=True)
