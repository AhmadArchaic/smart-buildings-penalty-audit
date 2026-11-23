"""Per-step split of a policy's penalty by building state. Usage: trace.py MODULE:CLASS BUILDING WEATHER STEPS"""
import os, sys, json, importlib
sys.path[:0] = [os.environ.get('SB_D', 'smart-buildings-starter-kit'), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'controllers')]
from pathlib import Path
import numpy as np
from smart_buildings import evaluate as ev
from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
spec, b, w, steps = sys.argv[1:5]
mod, cls = spec.split(':'); P = getattr(importlib.import_module(mod), cls)
d = Path(f'smart_buildings/training_scenarios/{b}/{w}')
env = ev.load_hybrid_environment(d / 'config.gin')
tr = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
env.metrics_path = None; env._occupancy_normalization_constant = 125.0
md = ev.create_building_meta_data(env, d)
pol = ev.initialize_policy(P, md, env.action_spec())
names = list(md['observation_field_names']); io = names.index('num_occupants'); isoon = names.index('comfort_mode_soon')
tr.reset(); obs = tr.last_obs; states = []
for _ in range(int(steps)):
    occ = float(obs[io]) > 0.5; soon = float(obs[isoon]) > 0.5
    states.append('occupied' if occ else ('pre-arrival' if soon else 'empty'))
    obs, *_ = tr.step(pol.get_action(obs))
comp = tr.reward_components
out = {}
for s, c in zip(states, comp):
    o = out.setdefault(s, [0, 0., 0., 0., 0., 0.])
    o[0] += 1; o[1] += 0.6 * -c['normalized_productivity_regret']; o[2] += 0.2 * c['normalized_energy_cost']; o[3] += 0.2 * c['normalized_carbon_emission']; o[4] += c['electricity_energy_cost']; o[5] += c['natural_gas_energy_cost']
n = len(comp); tot = sum(v[1] + v[2] + v[3] for v in out.values())
for s, (k, cf, en, ca, el, gs) in out.items():
    print(f'{s:12s} steps {k/n*100:5.1f}%  penalty share {(cf+en+ca)/tot*100:5.1f}%  comfort {cf/n:.5f} energy {en/n:.5f} carbon {ca/n:.5f}  $elec {el:.1f} $gas {gs:.1f} (gas share {gs/max(el+gs,1e-9)*100:.0f}%)')
print('total', round(tot / n, 5))
