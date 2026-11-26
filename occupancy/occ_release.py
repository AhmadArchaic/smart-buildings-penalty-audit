"""CtrlX + per-zone afternoon release.

mode "est":    legal. Ranks zones by an occupancy estimate built from the zone's own
               temperature, damper and reheat readings; releases the n_zones - ceil(N)
               lowest-ranked zones (N = building headcount, so at least that many are empty).
mode "oracle": analysis only. Releases a zone once the true zone occupancy (last interval)
               has been 0 for 3 steps after someone was there today. Needs ORACLE_OCC set.
mode "random": same count as "est", random zones (control for the ranking).
Released zones get CtrlX's standby band (relax_K None) or their low edge lowered by relax_K.
"""
import json, math, os
import numpy as np
from ctrl_x import CtrlX

ORACLE_OCC = None  # zone-ordered occupancy of the last completed interval (runner sets it)
_W = os.environ.get("OCC_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "occ_weights.json"))
SIGS = ("zone_air_temperature_sensor", "supply_air_damper_percentage_command", "reheat_valve_percentage_command")


class OccRelease(CtrlX):
    OCC = {"mode": "est", "relax_K": None, "start_hour": 13.0, "seed": 0, "weights": None, "thr": None}

    def __init__(self, building_meta_data, action_spec=None):
        super().__init__(building_meta_data, action_spec)
        o = self.OCC
        self.occ_mode, self.occ_relax, self.occ_start = o["mode"], o["relax_K"], float(o["start_hour"])
        w = o["weights"] or json.load(open(_W))
        self.occ_w, self.occ_b = np.asarray(w["w"], float), float(w["b"])
        fi = self.field_index
        self.occ_idx = [np.array([fi[f"{z}_{s}"] for z in self.zone_ids]) for s in SIGS]
        self.occ_n_idx = fi.get("num_occupants")
        self.occ_rng = np.random.default_rng(o["seed"])
        self._occ_reset_all()

    def reset(self):
        super().reset()
        if hasattr(self, "occ_idx"):
            self._occ_reset_all()

    def _occ_reset_all(self):
        z, s = self.n_zones, len(SIGS)
        self.o_s1 = np.zeros((s, z)); self.o_s2 = np.zeros((s, z)); self.o_k = 0
        self.o_b1 = np.zeros((s, z)); self.o_b2 = np.zeros((s, z)); self.o_bk = 0
        self.o_hist = []; self.o_score = []; self.o_reheat = []
        self.o_prev_hour = None
        self.o_released = np.zeros(z, bool)
        self.o_seen = np.zeros(z, bool); self.o_empty = np.zeros(z, int)

    # ---------------------------------------------------------------- estimator
    def _occ_update(self, obs, hour):
        if self.o_prev_hour is not None and hour < self.o_prev_hour:          # new day
            self.o_s1 += self.o_b1; self.o_s2 += self.o_b2; self.o_k += self.o_bk
            self.o_b1[:] = 0; self.o_b2[:] = 0; self.o_bk = 0
            self.o_released[:] = False; self.o_seen[:] = False; self.o_empty[:] = 0
            self.o_score = []
        self.o_prev_hour = hour
        N = float(obs[self.occ_n_idx]) if self.occ_n_idx is not None else 0.0
        X = np.stack([obs[i].astype(float) for i in self.occ_idx])
        X = X - np.median(X, axis=1, keepdims=True)
        if self.o_k > 12:
            mu = self.o_s1 / self.o_k
            sd = np.sqrt(np.maximum(self.o_s2 / self.o_k - mu ** 2, 0.0)) + 1e-6
            D = np.clip((X - mu) / np.maximum(sd, 1e-3), -5, 5)
        else:
            D = np.zeros_like(X)
        if 13.0 <= hour < 19.0 and N > 0:
            self.o_b1 += X; self.o_b2 += X ** 2; self.o_bk += 1
        self.o_hist.append(D); self.o_hist = self.o_hist[-13:]
        self.o_reheat.append(float(np.mean(obs[self.occ_idx[2]]))); self.o_reheat = self.o_reheat[-288:]
        H = self.o_hist
        tr6 = D - H[-7] if len(H) >= 7 else D
        m12 = np.mean(H[-12:], axis=0)
        F = np.concatenate([np.stack([D[s], tr6[s], m12[s]], -1) for s in range(len(SIGS))], -1)   # zone x 9
        heat = float(np.sum(self.o_reheat) / 288.0 > 0.02)
        F = np.concatenate([F, F * heat], -1)
        p = 1.0 / (1.0 + np.exp(-(F @ self.occ_w + self.occ_b)))
        self.o_score.append(p); self.o_score = self.o_score[-6:]
        return N

    def _occ_mask(self, obs, now):
        hour = self._hour(obs)
        N = self._occ_update(obs, hour)
        if self.occ_mode == "oracle" and ORACLE_OCC is not None:
            occ = np.asarray(ORACLE_OCC, float) > 0
            self.o_seen |= occ
            self.o_empty = np.where(occ, 0, self.o_empty + 1)
        if not now or hour < self.occ_start:
            return None
        wd = self._weekday(obs)
        if wd is not None and wd >= 5:
            return None
        if self.occ_mode == "oracle":
            self.o_released |= self.o_seen & (self.o_empty >= 3)
        elif N > 0:
            need = self.n_zones - int(math.ceil(N))
            have = int(self.o_released.sum())
            if need > have:
                free = np.flatnonzero(~self.o_released)
                if self.occ_mode == "random":
                    pick = self.occ_rng.choice(free, need - have, replace=False)
                else:
                    s = np.mean(self.o_score, axis=0)[free]
                    order = np.argsort(s)[:need - have]
                    thr = self.OCC.get("thr")
                    if thr is not None:
                        order = order[s[order] < thr]
                    pick = free[order]
                self.o_released[pick] = True
        return self.o_released if self.o_released.any() else None

    def _band(self, obs, temperature):
        low, high, now = super()._band(obs, temperature)
        mask = self._occ_mask(obs, now)
        if mask is None or self._released:
            return low, high, now
        if self.occ_relax is None:
            rl, rh = self._standby_band(temperature)
        else:
            rl, rh = low - float(self.occ_relax), high + float(self.occ_relax)
        return np.where(mask, rl, low), np.where(mask, rh, high), now


def variant(name, **kw):
    return type(name, (OccRelease,), {"OCC": {**OccRelease.OCC, **kw}})
