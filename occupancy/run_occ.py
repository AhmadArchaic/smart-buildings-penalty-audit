"""Usage: run_occ.py VARIANT BUILDING/WEATHER STEPS   (VARIANT: ctrlx | oracle | oracleK | est | estK | random)
Prints: variant, scenario, steps, score, comfort, energy, carbon"""
import os, sys, re
SB = os.environ.get("SB_D", "smart-buildings-starter-kit")
POL = os.environ.get("SB_POL", "sb26-stop-heating/policies")
sys.path[:0] = [SB, POL, os.path.dirname(os.path.abspath(__file__))]
os.chdir(SB)
from pathlib import Path
import numpy as np
from smart_buildings import evaluate as ev
from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
import occ_release as orl
from ctrl_x import CtrlX

V = {"ctrlx": CtrlX,
     "oracle": orl.variant("Oracle", mode="oracle"),
     "oracle1K": orl.variant("Oracle1K", mode="oracle", relax_K=1.0),
     "est": orl.variant("Est", mode="est"),
     "est1K": orl.variant("Est1K", mode="est", relax_K=1.0),
     "random": orl.variant("Random", mode="random"),
     "random1K": orl.variant("Random1K", mode="random", relax_K=1.0)}

def run(name, scen, steps):
    d = Path("smart_buildings/training_scenarios") / scen
    env = ev.load_hybrid_environment(d / "config.gin")
    tr = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    env.metrics_path = None; env._occupancy_normalization_constant = 125.0
    orig = env.get_reward_info_and_response
    num = lambda s: int(re.findall(r"\d+", s)[-1])
    def hooked():
        ri, rr = orig()
        z = ri.zone_reward_infos
        orl.ORACLE_OCC = [z[k].average_occupancy for k in sorted(z, key=num)]
        return ri, rr
    env.get_reward_info_and_response = hooked
    orl.ORACLE_OCC = None
    md = ev.create_building_meta_data(env, d)
    ev.run_policy(env, tr, ev.initialize_policy(V[name], md, env.action_spec()), steps)
    r = np.array(tr.rewards, float); c = tr.reward_components or []
    m = lambda k: float(np.mean([x.get(k, 0) for x in c])) if c else None
    row = [name, scen, len(r), float(-r.mean()), m("normalized_productivity_regret"), m("normalized_energy_cost"), m("normalized_carbon_emission")]
    print(row, flush=True)
    return row

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]))
