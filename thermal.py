"""Lumped (single-node) pack thermal model shared by Battery (ECM) and SPMBattery.

    C_th * dT/dt = Q_gen - hA * (T - T_amb)

T is kept in kelvin internally; the batteries report it in degC as 'T_batt'.
C_th and hA are placeholders: C_th ~ 2880 cells x 45 g x 1 kJ/(kg K) = 1.3e5 J/K,
hA = 30 W/K is a guess for a naturally/lightly cooled pack. Replace with real
pack data, or sweep them (a cold or hot ambient matters more than exact values).
"""


class LumpedThermal:
    def __init__(self, dt, T_init=298.15, T_amb=None, C_th=1.3e5, hA=30.0):
        self.dt = dt
        self.T = float(T_init)                       # [K]
        self.T_amb = float(T_init if T_amb is None else T_amb)
        self.C_th = C_th                             # [J/K]
        self.hA = hA                                 # [W/K]
        self.T_max = self.T
        self.T_min = self.T

    def step(self, Q_gen):
        """Advance one step with heat generation Q_gen [W]; returns T in degC."""
        self.T += self.dt * (Q_gen - self.hA * (self.T - self.T_amb)) / self.C_th
        self.T_max = max(self.T_max, self.T)
        self.T_min = min(self.T_min, self.T)
        return self.T - 273.15