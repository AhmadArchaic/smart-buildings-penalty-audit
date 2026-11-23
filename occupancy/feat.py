import sys, re, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
obs, occ, fields = z['obs'], z['occ'][::2], list(z['fields'])
import re as _re; _z = list(z['zones']); occ = occ[:, sorted(range(len(_z)), key=lambda i: int(_re.findall(r'\d+', _z[i])[-1]))]
rooms = sorted(list(z['zones']), key=lambda s: int(re.findall(r'\d+', s)[-1]))
n = min(len(obs) - 1, len(occ))
O = occ[:n]                      # occupancy during step t -> visible in obs[t+1]
hour = ((np.arange(n) + 1) / 12 + 6) % 24
work = (O.sum(1) > 0) & (hour >= 13) & (hour < 19)
def auc(x, y):
    from scipy.stats import rankdata; r = rankdata(x) - 1; n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
for feat in ['reheat_valve_percentage_command', 'supply_air_damper_percentage_command', 'supply_air_flowrate_setpoint', 'zone_air_temperature_sensor']:
    cols = [fields.index(f'VAV_{int(re.findall(r"\d+", r)[-1])}_{feat}') for r in rooms]
    X = obs[1:n + 1, cols][work]
    X = X - np.median(X, axis=1, keepdims=True)          # common mode per step
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)               # per zone
    y = (O[work] > 0)
    h = np.floor(hour[work]).astype(int)
    per = [auc(X[h == k].ravel(), y[h == k].ravel()) for k in np.unique(h) if 0 < y[h == k].mean() < 1]
    print(f'{feat:42s} AUC {auc(X.ravel(), y.ravel()):.3f}  within-hour mean {np.mean(per):.3f}')
