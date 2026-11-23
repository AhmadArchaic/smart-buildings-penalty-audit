"""Occupancy-aware setback on top of the shipped pid_water controller.

Idea: comfort is occupancy-weighted, so an empty building costs nothing in
comfort. While nobody is there (nights, weekends) we widen the band to a frost
floor / heat ceiling and let the AHUs idle. Shortly before the next workday
starts we go back to the normal band early enough to pre-heat/pre-cool.

Everything is read from the observation and building metadata -- no weather
label, building id or scenario lookup.
"""

import numpy as np

from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy


class SetbackPolicy(PIDWaterResetPolicy):
    # Band used while the building is empty and no preheat is due (K).
    UNOCC_LOW = 283.0
    UNOCC_HIGH = 301.0
    # Preheat lead time before the first arrivals: base + per-K of deficit (h).
    LEAD_BASE_H = 1.0
    LEAD_PER_K_H = 0.5
    LEAD_MAX_H = 5.0
    # Occupancy count at/below which the building counts as empty.
    EMPTY_OCC = 0.5

    @property
    def name(self) -> str:
        return "setback"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        names = list(self.field_names)
        self._occ_idx = names.index("num_occupants") if "num_occupants" in names else None
        self._sched_comfort = tuple(self.comfort_temp_window)
        self._sched_eco = tuple(self.eco_temp_window)

    # -- helpers -----------------------------------------------------------
    def _unocc_low(self, obs) -> float:
        return self.UNOCC_LOW

    def _hours_to_next_start(self, obs) -> float:
        angle = np.arctan2(obs[self.hod_sin_index], obs[self.hod_cos_index]) % (2 * np.pi)
        hour = angle / (2 * np.pi) * 24.0
        _, day = self._get_hour_and_day(obs)
        for d in range(0, 8):
            if (day + d) % 7 < 5:
                t = d * 24.0 + self.morning_start_hour - hour
                if t > 0:
                    return t
        return 24.0

    def _mean_zone_temp(self, obs) -> float:
        idx = [j for c in self.ahu_controllers for j in c["zone_obs_indices"]]
        return float(np.mean([self._denormalize_zone_temp(obs[j]) for j in idx]))

    # -- control -----------------------------------------------------------
    def get_action(self, obs):
        occ = float(obs[self._occ_idx]) if self._occ_idx is not None else 1.0
        if occ > self.EMPTY_OCC:
            band = None  # occupied: schedule band, exactly like pid_water
        else:
            t_mean = self._mean_zone_temp(obs)
            lo, hi = self._sched_comfort
            deficit = max(lo - t_mean, t_mean - hi, 0.0)
            lead = min(self.LEAD_BASE_H + self.LEAD_PER_K_H * deficit, self.LEAD_MAX_H)
            if self._hours_to_next_start(obs) <= lead:
                band = (lo, hi)  # preheat / precool toward the workday band
            else:
                band = (self._unocc_low(obs), self.UNOCC_HIGH)

        if band is None:
            return super().get_action(obs)

        # The parent derives the band from the hour via these two attributes
        # (the fixed_setpoint baseline uses the same trick); pin both.
        saved = self.comfort_temp_window, self.eco_temp_window
        self.comfort_temp_window = self.eco_temp_window = band
        try:
            return super().get_action(obs)
        finally:
            self.comfort_temp_window, self.eco_temp_window = saved


class Combo(SetbackPolicy):
    """Best-of so far: heating floor set back to UNOCC_LOW while empty, cooling
    ceiling kept at the schedule's eco high (a higher ceiling hurt phoenix),
    and separate pre-conditioning leads for heating vs cooling (an early
    preheat cost more energy than it saved comfort in fairbanks)."""

    LEAD_HEAT_BASE_H = 0.5
    LEAD_HEAT_PER_K_H = 0.25
    LEAD_HEAT_MAX_H = 3.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.UNOCC_HIGH = self._sched_eco[1]

    def get_action(self, obs):
        occ = float(obs[self._occ_idx]) if self._occ_idx is not None else 1.0
        if occ > self.EMPTY_OCC:
            return PIDWaterResetPolicy.get_action(self, obs)
        lo, hi = self._sched_comfort
        t_mean = self._mean_zone_temp(obs)
        if t_mean < lo:   # needs heating
            lead = min(self.LEAD_HEAT_BASE_H + self.LEAD_HEAT_PER_K_H * (lo - t_mean), self.LEAD_HEAT_MAX_H)
        else:             # needs cooling (or in band)
            lead = min(self.LEAD_BASE_H + self.LEAD_PER_K_H * max(t_mean - hi, 0.0), self.LEAD_MAX_H)
        if self._hours_to_next_start(obs) <= lead:
            band = (lo, hi)
        else:
            band = (self._unocc_low(obs), self.UNOCC_HIGH)
        saved = self.comfort_temp_window, self.eco_temp_window
        self.comfort_temp_window = self.eco_temp_window = band
        try:
            return PIDWaterResetPolicy.get_action(self, obs)
        finally:
            self.comfort_temp_window, self.eco_temp_window = saved


class SetbackShortLead(SetbackPolicy):
    """Same setback, shorter pre-conditioning ramp (early start costs energy
    in deep cold)."""

    LEAD_BASE_H = 0.5
    LEAD_PER_K_H = 0.25
    LEAD_MAX_H = 3.0


class SetbackCool(SetbackPolicy):
    """Cooling-side setback only: empty-building heating floor stays at the
    schedule's eco low, because in deep cold the morning recovery costs more
    than the night saves (measured on fairbanks)."""

    def _unocc_low(self, obs) -> float:
        return self._sched_eco[0]


class LeadOnly(SetbackPolicy):
    """No deeper setback at all: empty building held to the schedule's eco
    band. Only the early pre-conditioning before arrivals differs from
    pid_water."""

    def _unocc_low(self, obs) -> float:
        return self._sched_eco[0]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.UNOCC_HIGH = self._sched_eco[1]


class SetbackOAT(SetbackPolicy):
    """Heating floor deepens only when it is not very cold outside:
    eco low at OAT <= 263 K, UNOCC_LOW at OAT >= 278 K, linear between."""

    OAT_COLD_K = 263.0
    OAT_MILD_K = 278.0

    def _unocc_low(self, obs) -> float:
        oat = [float(obs[c["oat_idx"]]) for c in self.ahu_controllers if c.get("oat_idx") is not None]
        if not oat:
            return self._sched_eco[0]
        w = np.clip((np.mean(oat) - self.OAT_COLD_K) / (self.OAT_MILD_K - self.OAT_COLD_K), 0.0, 1.0)
        return float(self._sched_eco[0] + w * (self.UNOCC_LOW - self._sched_eco[0]))
