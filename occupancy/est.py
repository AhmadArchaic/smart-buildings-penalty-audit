"""Combined per-zone occupancy estimator, causal features, leave-one-scenario-out.
Usage: est.py a.npz b.npz ...   Reports AUC and the release-by-headcount test."""
import sys, re, numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

num = lambda s: int(re.findall(r'\d+', s)[-1])
SIG = ['zone_air_temperature_sensor', 'supply_air_damper_percentage_command', 'reheat_valve_percentage_command']

def auc(x, y):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    r = rankdata(x) - 1
    return (r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)

def load(path):
    z = np.load(path, allow_pickle=True)
    f = list(z['fields']); zs = list(z['zones'])
    occ = z['occ'][::2][:, sorted(range(len(zs)), key=lambda i: num(zs[i]))]
    rooms = sorted(zs, key=num)
    obs = z['obs']; n = min(len(obs) - 1, len(occ))
    O = occ[:n]
    nxt = obs[1:n + 1]
    hour = (np.arctan2(nxt[:, f.index('hod_sin_000')], nxt[:, f.index('hod_cos_000')]) % (2 * np.pi)) * 12 / np.pi
    feats = []
    Nobs = nxt[:, f.index('num_occupants')]
    RH = nxt[:, [f.index(f'VAV_{num(r)}_reheat_valve_percentage_command') for r in rooms]].mean(1)
    for s in SIG:
        X = nxt[:, [f.index(f'VAV_{num(r)}_{s}') for r in rooms]]
        X = X - np.median(X, axis=1, keepdims=True)                     # building common mode
        # causal per-zone z-score: mean/std of this zone over all earlier afternoons (13-19 h)
        aft = (hour >= 13) & (hour < 19) & (Nobs > 0)
        D = np.zeros_like(X); s1 = np.zeros(X.shape[1]); s2 = np.zeros(X.shape[1]); k = 0
        day = np.cumsum(np.r_[0, (np.diff(hour) < 0)])
        cur = 0; b1 = np.zeros(X.shape[1]); b2 = np.zeros(X.shape[1]); bk = 0
        for t in range(n):
            if day[t] != cur:                      # roll the finished day's afternoon into the stats
                s1 += b1; s2 += b2; k += bk; b1[:] = 0; b2[:] = 0; bk = 0; cur = day[t]
            if k > 12:
                mu = s1 / k; sd = np.sqrt(np.maximum(s2 / k - mu ** 2, 0)) + 1e-6
                D[t] = np.clip((X[t] - mu) / np.maximum(sd, 1e-3), -5, 5)
            if aft[t]:
                b1 += X[t]; b2 += X[t] ** 2; bk += 1
        tr6 = np.vstack([np.zeros((6, D.shape[1])), D[6:] - D[:-6]])
        cs = np.cumsum(D, 0); m12 = (cs - np.vstack([np.zeros((12, D.shape[1])), cs[:-12]])) / 12
        feats += [D, tr6, m12]
    N = nxt[:, f.index('num_occupants')] if 'num_occupants' in f else O.sum(1)
    true_N = O.sum(1)
    F = np.stack(feats, -1)                                             # t, zone, feat
    heat = (np.convolve(RH, np.ones(288) / 288, 'full')[:n] > 0.02).astype(float)[:, None, None]   # building in heating mode
    F = np.concatenate([F, F * heat], -1)
    return dict(F=F, O=O, hour=hour, trueN=true_N, rooms=rooms)

data = {p: load(p) for p in sys.argv[1:]}
def mask(d): return (d['trueN'] > 0) & (d['hour'] >= 13) & (d['hour'] < 19) & (np.arange(len(d['hour'])) >= 288)
for test in data:
    tr = [d for p, d in data.items() if p != test]
    Xtr = np.concatenate([d['F'][mask(d)].reshape(-1, d['F'].shape[-1]) for d in tr])
    ytr = np.concatenate([(d['O'][mask(d)] > 0).ravel() for d in tr])
    clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    d = data[test]; m = mask(d)
    p = clf.predict_proba(d['F'][m].reshape(-1, d['F'].shape[-1]))[:, 1].reshape(m.sum(), -1)
    y = d['O'][m] > 0
    h = np.floor(d['hour'][m]).astype(int)
    wh = np.mean([auc(p[h == k].ravel(), y[h == k].ravel()) for k in np.unique(h) if 0 < y[h == k].mean() < 1])
    # release test: occupants present = trueN (people). Occupied zones <= people; release zones-people? use count of
    # occupied zones unknown; release all zones except the top-k scored, k = ceil(people) (upper bound on occupied zones)
    occ_people = d['O'][m]
    miss_people = []; released = []
    for t in range(m.sum()):
        k = int(np.ceil(occ_people[t].sum()))
        keep = np.argsort(-p[t])[:k]
        rel = np.setdiff1d(np.arange(p.shape[1]), keep)
        miss_people.append(occ_people[t, rel].sum() / max(occ_people[t].sum(), 1e-9))
        released.append(len(rel) / p.shape[1])
    # random-ranking baseline
    rng = np.random.default_rng(0); miss_rand = []
    for t in range(m.sum()):
        k = int(np.ceil(occ_people[t].sum())); keep = rng.permutation(p.shape[1])[:k]
        rel = np.setdiff1d(np.arange(p.shape[1]), keep)
        miss_rand.append(occ_people[t, rel].sum() / max(occ_people[t].sum(), 1e-9))
    print(f'{test:16s} AUC {auc(p.ravel(), y.ravel()):.3f} within-hour {wh:.3f} | zones released {np.mean(released):.2f} '
          f'| share of present people in released zones: model {np.mean(miss_people):.3f} random {np.mean(miss_rand):.3f}')
