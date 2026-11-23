import os, sys, glob, re, csv
from pathlib import Path
from multiprocessing import Pool
D = os.environ.get("SB_D") or next(str(p.parent) for p in [*Path("/kaggle/input").glob("*/smart_buildings"), *Path("/kaggle/input").glob("*/*/smart_buildings")] if "smart-buildings" in str(p))
POL = os.environ.get("SB_POL") or os.path.dirname(sorted(glob.glob("/kaggle/input/**/policies/ctrl_x.py", recursive=True))[0])
STEPS = int(os.environ.get("SB_STEPS", 1728))  # 6 days
FEATS = ["zone_air_temperature_sensor", "supply_air_damper_percentage_command", "reheat_valve_percentage_command"]
COLS = ["scenario", "zones", "afternoon_zone_steps", "occupied_frac"] + \
       [f"{k}_{s}" for k in ("temp", "damper", "reheat") for s in ("auc", "auc_within_hour")] + \
       ["arx_auc_w1", "arx_auc_w6", "arx_auc_w12", "arx_auc_w6_within_hour"]

def auc(x, y):
    import numpy as np
    from scipy.stats import rankdata
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(x) - 1
    return float((r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))

def within(X, y, h):
    import numpy as np
    v = [auc(X[h == k].ravel(), y[h == k].ravel()) for k in np.unique(h)]
    v = [a for a in v if a == a]
    return float(np.mean(v)) if v else float("nan")

def run(scen):
    sys.path[:0] = [D, POL]; os.chdir(D)
    import numpy as np
    from smart_buildings import evaluate as ev
    from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
    from ctrl_x import CtrlX
    d = Path("smart_buildings/training_scenarios") / scen
    env = ev.load_hybrid_environment(d / "config.gin")
    tr = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    env.metrics_path = None; env._occupancy_normalization_constant = 125.0
    occ_log = []; orig = env.get_reward_info_and_response
    def hooked():
        ri, rr = orig(); occ_log.append({k: v.average_occupancy for k, v in ri.zone_reward_infos.items()}); return ri, rr
    env.get_reward_info_and_response = hooked
    md = ev.create_building_meta_data(env, d)
    pol = ev.initialize_policy(CtrlX, md, env.action_spec())
    tr.reset(); obs = tr.last_obs; O_list, obs_list = [], []
    for _ in range(STEPS):
        obs_list.append(np.asarray(obs, float))
        n0 = len(occ_log)
        obs, *_ = tr.step(ev.validate_action(ev.get_policy_action(pol, obs), env.action_spec()))
        O_list.append(occ_log[-1] if len(occ_log) > n0 else {})
    obs = np.array(obs_list + [np.asarray(obs, float)])
    f = [str(x) for x in md["observation_field_names"]]
    num = lambda s: int(re.findall(r"\d+", s)[-1])
    rooms = sorted([k for k in O_list[-1] if re.search(r"\d", k)], key=num)
    rooms = [r for r in rooms if all(f"VAV_{num(r)}_{ft}" in f for ft in FEATS)]
    O = np.array([[o.get(r, 0.0) for r in rooms] for o in O_list])          # occupancy during step t
    n = len(O)
    hour = (np.arctan2(obs[1:n + 1, f.index("hod_sin_000")], obs[1:n + 1, f.index("hod_cos_000")]) % (2 * np.pi)) * 12 / np.pi
    work = (O.sum(1) > 0) & (hour >= 13) & (hour < 19)
    y = O[work] > 0; h = np.floor(hour[work]).astype(int)
    row = [scen, len(rooms), int(work.sum() * len(rooms)), float(y.mean())]
    get = lambda ft: obs[:, [f.index(f"VAV_{num(r)}_{ft}") for r in rooms]]
    for ft in FEATS:
        X = get(ft)[1:n + 1][work]
        X = X - np.median(X, axis=1, keepdims=True)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        row += [auc(X.ravel(), y.ravel()), within(X, y, h)]
    T, X1, X2 = get(FEATS[0]), get(FEATS[1]), get(FEATS[2])
    dT = T[1:n + 1] - T[:n]; cm = np.median(dT, axis=1); res = np.zeros_like(dT)
    for j in range(T.shape[1]):
        A = np.column_stack([np.ones(n), T[:n, j], T[:n, j] - np.median(T[:n], 1), X1[:n, j], X2[:n, j], cm,
                             X1[:n, j] * T[:n, j], np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)])
        w, *_ = np.linalg.lstsq(A, dT[:, j], rcond=None); res[:, j] = dT[:, j] - A @ w
    for win in (1, 6, 12):
        S = np.apply_along_axis(lambda v: np.convolve(v, np.ones(win), "full")[:n], 0, res)
        row.append(auc(S[work].ravel(), y.ravel()))
        if win == 6: S6 = S[work]
    row.append(within(S6, y, h))
    print(row, flush=True); return row

def safe(scen):
    try: return run(scen)
    except Exception as e:
        print("FAIL", scen, repr(e), flush=True); return [scen] + ["nan"] * (len(COLS) - 1)

if __name__ == "__main__":
    base = Path(D) / "smart_buildings/training_scenarios"
    scens = sorted(str(p.parent.relative_to(base)) for p in base.glob("*/*/config.gin"))
    print(len(scens), "scenarios; round2 simulator files present:", os.path.exists(os.path.join(D, "smart_buildings", "check_return_air.py")), flush=True)
    with open("/kaggle/working/occupancy_detection_30.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(COLS)
        with Pool(os.cpu_count()) as pool:
            for row in pool.imap_unordered(safe, scens): w.writerow(row); fh.flush()
    print("DONE", flush=True)
