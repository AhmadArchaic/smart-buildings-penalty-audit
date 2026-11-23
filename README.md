# Where the Penalty Goes: NeurIPS 2026 Smart Buildings Challenge

Code, results and paper source for *Where the Penalty Goes: A Climate-Resolved Audit of Transferable HVAC Control in the Smart Buildings Challenge* (Ahmad Dweikat, Princess Sumaya University for Technology).

The paper takes the challenge score apart by reward term, climate and building state for a weak baseline (`pid_water`) and a strong public controller (CtrlX), shows that CtrlX sits at an optimum, and tests whether per-zone occupancy, the largest remaining lever, can be inferred and used. Every number comes from the Round 2 simulator.

## Layout

| Folder | What is in it |
|---|---|
| `submission/loader.py` | The Kaggle submission (public score 0.00983): CtrlX wrapped for the inference server, with action clipping and a safe fallback. |
| `controllers/` | Our schedule variants of `pid_water` (Setback, LeadOnly, Combo) and the CtrlX parameter variants used in the sweep. |
| `experiments/kaggle/` | Kaggle notebook scripts: the 30-scenario decomposition, the CtrlX sweep, the 30-scenario occupancy detection test. |
| `experiments/local/` | Run one policy on one scenario; split a run's penalty by building state (occupied / pre-arrival / empty). |
| `occupancy/` | Occupancy logging, detection tests (`feat.py`, `arx.py`), the estimator (`est.py`, `occ_weights.json`) and the afternoon release controller (`occ_release.py`, `run_occ.py`). |
| `results/` | Per-scenario CSVs behind every table and figure. |
| `paper/` | LaTeX source, figures and the scripts that draw them. |

## Requirements

* The challenge starter kit with the Round 2 fixes (set `SB_D` to its folder): <https://www.kaggle.com/competitions/neurips-2026-smart-buildings>
* CtrlX by S. Morev, Apache-2.0 (set `SB_POL` to its `policies` folder): <https://github.com/MorS1337/sb26-stop-heating>
* Python 3.10+, numpy, scipy, scikit-learn, matplotlib.

The simulator is deterministic, so every run reproduces bit for bit.

## Examples

```bash
export SB_D=/path/to/starter-kit SB_POL=/path/to/sb26-stop-heating/policies

# CtrlX vs the true-occupancy release vs our estimator, 7 days in Phoenix
python occupancy/run_occ.py ctrlx  building_train_1/phoenix_5 2016
python occupancy/run_occ.py oracle building_train_1/phoenix_5 2016
python occupancy/run_occ.py est1K  building_train_1/phoenix_5 2016
```

On Kaggle, attach the competition data and the CtrlX dataset to a CPU notebook and run a script from `experiments/kaggle/`; each writes a CSV to `/kaggle/working`.

## Main results

* CtrlX beats `pid_water` in all 30 training scenarios (median 8.3x); what CtrlX still pays is energy and carbon, mostly gas for heating in cold climates while the building is occupied.
* Schedule ideas worth up to 24% on `pid_water` are worth at most 0.1% on CtrlX; no single parameter change improves it.
* A person leaves only a weak per-zone trace in the observations (mean best AUC 0.63). Releasing empty zones with true occupancy gains 5.3%; with our estimator it loses in all 30 scenarios.

## License

MIT for the code in this repository (see `LICENSE`). CtrlX and Wu's controller are not included and remain under their own Apache-2.0 licenses.
