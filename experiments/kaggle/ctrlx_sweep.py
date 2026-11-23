import os, sys, glob, time, csv
from pathlib import Path
from multiprocessing import Pool
D = next(str(p.parent) for p in [*Path("/kaggle/input").glob("*/smart_buildings"), *Path("/kaggle/input").glob("*/*/smart_buildings")] if "smart-buildings" in str(p))
POL = os.path.dirname(sorted(glob.glob("/kaggle/input/**/policies/ctrl_x.py", recursive=True))[0])
V = {"CtrlX": {}, "Precool": {"early_precool_enabled": True}, "LongLead": {"arrival_lead_hours": 1.0},
     "FreePrecool": {"precool_free_margin_K": 0.5}, "Margin04": {"comfort_margin": 0.04}, "Margin12": {"comfort_margin": 0.12},
     "HeatResp65": {"heating_response_scale": 0.65}, "HeatResp85": {"heating_response_scale": 0.85},
     "WaterCeil345": {"water_ceiling": 345.0}, "WaterFloor325": {"water_floor": 325.0}}

def run(job):
    name, scen, steps = job
    sys.path[:0] = [D, POL]; os.chdir(D)
    import numpy as np
    from smart_buildings import evaluate as ev
    from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
    from ctrl_x import CtrlX
    P = type(name, (CtrlX,), {"PARAMS": {**CtrlX.PARAMS, **V[name]}})
    d = Path("smart_buildings/training_scenarios") / scen
    env = ev.load_hybrid_environment(d / "config.gin")
    tr = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    env.metrics_path = None; env._occupancy_normalization_constant = 125.0
    md = ev.create_building_meta_data(env, d)
    ev.run_policy(env, tr, ev.initialize_policy(P, md, env.action_spec()), steps)
    r = np.array(tr.rewards, float); c = tr.reward_components or []
    m = lambda k: float(np.mean([x.get(k, 0) for x in c])) if c else None
    row = [name, scen, len(r), float(-r.mean()), m("normalized_productivity_regret"), m("normalized_energy_cost"), m("normalized_carbon_emission")]
    print(row, flush=True); return row

if __name__ == "__main__":
    four = ["building_train_1/" + w for w in ("phoenix_5", "denver_7", "fairbanks_1", "singapore_8")]
    jobs = [(v, s, 2016) for v in V for s in four]
    print(len(jobs), "jobs", flush=True)
    with open("/kaggle/working/results_r2_variants.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["variant", "scenario", "steps", "score", "comfort", "energy", "carbon"])
        with Pool(os.cpu_count()) as pool:
            for row in pool.imap_unordered(run, jobs): w.writerow(row); f.flush()
    print("DONE", flush=True)
