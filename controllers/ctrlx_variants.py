import os, sys
sys.path.insert(0, os.environ.get("SB_POL", "sb26-stop-heating/policies"))
from ctrl_x import CtrlX
_BASE = dict(CtrlX.PARAMS)
CtrlX.PARAMS = {**CtrlX.PARAMS, "water_idle_hold": True, "water_peak_hold_hours": 30, "water_rise_guard_K": 2.5, "water_offpeak_floor": 0}

class Precool(CtrlX):
    PARAMS = {**CtrlX.PARAMS, "early_precool_enabled": True}

class LongLead(CtrlX):
    PARAMS = {**CtrlX.PARAMS, "arrival_lead_hours": 1.0}

class FreePrecool(CtrlX):
    PARAMS = {**CtrlX.PARAMS, "precool_free_margin_K": 0.5}

def _v(name, **kw):
    globals()[name] = type(name, (CtrlX,), {"PARAMS": {**CtrlX.PARAMS, **kw}})
_v("Margin04", comfort_margin=0.04)
_v("Margin12", comfort_margin=0.12)
_v("HeatResp65", heating_response_scale=0.65)
_v("HeatResp85", heating_response_scale=0.85)
_v("WaterCeil345", water_ceiling=345.0)
_v("WaterFloor325", water_floor=325.0)
_v("WaterFloor335", water_floor=335.0)

class NoTank(CtrlX):
    PARAMS = dict(_BASE)

class NT(NoTank): pass
def _w(name, **kw):
    globals()[name] = type(name, (NoTank,), {"PARAMS": {**_BASE, **kw}})
_w("Ident", ident_enabled=True)
_w("Gas07", gas_ratio_scale=0.7)
_w("Gas15", gas_ratio_scale=1.5)
_w("Gas20", gas_ratio_scale=2.0)
