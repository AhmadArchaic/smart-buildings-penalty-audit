"""Run one policy on one scenario for N steps; write summary json. Usage:
run_one.py MODULE:CLASS BUILDING WEATHER STEPS OUT.json"""
import os, sys, json, time, importlib
sys.path.insert(0, os.environ.get('SB_D', 'smart-buildings-starter-kit')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'controllers'))
from pathlib import Path
import numpy as np
from smart_buildings import evaluate as ev
from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker

spec, building, weather, steps, out = sys.argv[1:6]
mod, cls = spec.split(':')
policy_class = getattr(importlib.import_module(mod), cls)
d = Path(f'smart_buildings/training_scenarios/{building}/{weather}')
env = ev.load_hybrid_environment(d / 'config.gin')
tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
env.metrics_path = None
env._occupancy_normalization_constant = 125.0
md = ev.create_building_meta_data(env, d)
policy = ev.initialize_policy(policy_class, md, env.action_spec())
t0 = time.time()
r = ev.run_policy(env, tracker, policy, int(steps))
rewards = np.array(tracker.rewards, dtype=float)
comp = tracker.reward_components or []
res = dict(policy=spec, building=building, weather=weather, steps=len(rewards),
           score=float(-rewards.mean()),  # lower is better
           prod=float(np.mean([c.get('normalized_productivity_regret', 0) for c in comp])) if comp else None,
           energy=float(np.mean([c.get('normalized_energy_cost', 0) for c in comp])) if comp else None,
           carbon=float(np.mean([c.get('normalized_carbon_emission', 0) for c in comp])) if comp else None,
           secs=time.time() - t0)
json.dump(res, open(out, 'w'))
print(json.dumps(res))
