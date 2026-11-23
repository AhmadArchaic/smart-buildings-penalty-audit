"""Per-zone ARX residual as an occupancy detector. Usage: arx.py FILE.npz"""
import sys, re, numpy as np
from scipy.stats import rankdata
z = np.load(sys.argv[1], allow_pickle=True)
obs, occ, f = z['obs'], z['occ'][::2], list(z['fields'])
import re as _re; _z = list(z['zones']); occ = occ[:, sorted(range(len(_z)), key=lambda i: int(_re.findall(r'\d+', _z[i])[-1]))]
rooms = sorted(list(z['zones']), key=lambda s: int(re.findall(r'\d+', s)[-1]))
ids = [int(re.findall(r'\d+', r)[-1]) for r in rooms]
col = lambda name: [f.index(f'VAV_{i}_{name}') for i in ids]
T = obs[:, col('zone_air_temperature_sensor')]
X1 = obs[:, col('supply_air_damper_percentage_command')]
X2 = obs[:, col('reheat_valve_percentage_command')]
X3 = obs[:, col('supply_air_flowrate_setpoint')]
n = min(len(obs) - 1, len(occ))
O = occ[:n]
hour = ((np.arange(n) + 1) / 12 + 6) % 24
dT = T[1:n + 1] - T[:n]
cm = np.median(dT, axis=1)
def auc(x, y):
    r = rankdata(x) - 1; n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
res = np.zeros_like(dT); gain = []
for j in range(T.shape[1]):
    A = np.column_stack([np.ones(n), T[:n, j], T[:n, j] - np.median(T[:n], 1), X1[:n, j], X2[:n, j], X3[:n, j],
                         cm, X1[:n, j] * T[:n, j], np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)])
    w, *_ = np.linalg.lstsq(A, dT[:, j], rcond=None)
    res[:, j] = dT[:, j] - A @ w
    w2, *_ = np.linalg.lstsq(np.column_stack([A, O[:, j]]), dT[:, j], rcond=None)
    gain.append(w2[-1])
print('fitted occupant coefficient (normalized K/step per person): median %.5f  vs residual std %.5f' % (np.median(gain), res.std()))
work = (O.sum(1) > 0) & (hour >= 13) & (hour < 19)
for win in (1, 6, 12):
    k = np.ones(win)
    S = np.apply_along_axis(lambda v: np.convolve(v, k, 'full')[:n], 0, res)  # trailing sum
    y = O[work] > 0
    print(f'window {win:2d} steps: AUC occupied-vs-empty {auc(S[work].ravel(), y.ravel()):.3f}')
