"""Log obs + true per-zone occupancy for CtrlX. Usage: log_run.py BUILDING WEATHER STEPS OUT.npz"""
import os, sys, time
sys.path.insert(0, os.environ.get('SB_D', 'smart-buildings-starter-kit')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'controllers'))
from pathlib import Path
import numpy as np
from smart_buildings import evaluate as ev
from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
import ctrlx_local as cl
building, weather, steps, out = sys.argv[1:5]
d = Path(os.environ.get('SB_D', 'smart-buildings-starter-kit') + f'/smart_buildings/training_scenarios/{building}/{weather}')
env = ev.load_hybrid_environment(d / 'config.gin')
tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
env.metrics_path = None; env._occupancy_normalization_constant = 125.0
occ_log = []
orig = env.get_reward_info_and_response
def hooked():
    ri, rr = orig()
    occ_log.append({k: v.average_occupancy for k, v in ri.zone_reward_infos.items()})
    return ri, rr
env.get_reward_info_and_response = hooked
md = ev.create_building_meta_data(env, d)
policy = ev.initialize_policy(cl.NoTank, md, env.action_spec())
obs_log = []
tracker.reset(); obs = tracker.last_obs
t0 = time.time()
for i in range(int(steps)):
    obs_log.append(np.asarray(obs, float))
    a = ev.validate_action(ev.get_policy_action(policy, obs), env.action_spec())
    obs, *_ = tracker.step(a)
zones = sorted(occ_log[-1].keys())
occ = np.array([[o.get(z, np.nan) for z in zones] for o in occ_log])
np.savez(out, obs=np.array(obs_log), occ=occ, zones=np.array(zones), fields=np.array(md['observation_field_names']),
         rewards=np.array(tracker.rewards, float))
print('done', time.time() - t0, occ.shape, len(obs_log))
