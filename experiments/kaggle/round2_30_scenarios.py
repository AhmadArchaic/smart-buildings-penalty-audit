import os, sys, glob, time, csv
from pathlib import Path
from multiprocessing import Pool
D = next(str(p.parent) for p in [*Path("/kaggle/input").glob("*/smart_buildings"), *Path("/kaggle/input").glob("*/*/smart_buildings")] if "smart-buildings" in str(p))
POL = os.path.dirname(sorted(glob.glob("/kaggle/input/**/policies/ctrl_x.py", recursive=True))[0])
HERE = os.getcwd()
OUT = "/kaggle/working/results_round2.csv"
TANK = {"water_idle_hold": True, "water_peak_hold_hours": 30, "water_rise_guard_K": 2.5, "water_offpeak_floor": 0}

def run(job):
    pol, scen, steps = job
    sys.path[:0] = [D, POL, HERE]
    os.chdir(D)
    import numpy as np
    from smart_buildings import evaluate as ev
    from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
    if pol in ("CtrlX", "CtrlX_notank"):
        from ctrl_x import CtrlX
        P = type(pol, (CtrlX,), {"PARAMS": {**CtrlX.PARAMS, **(TANK if pol == "CtrlX" else {})}})
    elif pol == "pid_water":
        from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy as P
    else:
        import my_policy
        P = getattr(my_policy, pol)
    d = Path("smart_buildings/training_scenarios") / scen
    env = ev.load_hybrid_environment(d / "config.gin")
    tr = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    md = ev.create_building_meta_data(env, d)
    policy = ev.initialize_policy(P, md, env.action_spec())
    t0 = time.time()
    ev.run_policy(env, tr, policy, steps)
    r = np.array(tr.rewards, float); comp = tr.reward_components or []
    m = lambda k: float(np.mean([c.get(k, 0) for c in comp])) if comp else None
    row = [pol, scen, len(r), float(-r.mean()), m("normalized_productivity_regret"), m("normalized_energy_cost"), m("normalized_carbon_emission"), round(time.time() - t0)]
    print(row, flush=True)
    return row

if __name__ == "__main__":
    print("round2 simulator files present:", os.path.exists(os.path.join(D, "smart_buildings", "check_return_air.py")), flush=True)
    root = Path(D) / "smart_buildings/training_scenarios"
    scens = sorted(str(p.relative_to(root)) for p in root.glob("*/*") if p.is_dir())
    four = ["building_train_1/" + w for w in ("phoenix_5", "denver_7", "fairbanks_1", "singapore_8")]
    jobs = [(p, s, 4032) for p in ("SetbackPolicy", "LeadOnly", "Combo", "CtrlX_notank", "pid_water", "CtrlX") for s in four]
    jobs += [(p, s, 8640) for p in ("CtrlX", "pid_water") for s in scens]
    print(len(jobs), "jobs", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["policy", "scenario", "steps", "score", "comfort", "energy", "carbon", "secs"])
        with Pool(os.cpu_count()) as pool:
            for row in pool.imap_unordered(run, jobs):
                w.writerow(row); f.flush()
    print("DONE", flush=True)
