"""
Drop-in battery model built on SPPy's single particle model (SPM).
https://github.com/m0in92/SPPy   (licence: CC BY-NC 4.0, non-commercial use)

Same interface as main.Battery:
    step(P_motor, P_regen) -> dict(I_batt, net_power, V_terminal, SoC, T_batt)
Sign convention of THIS model (as in main.py): discharge current is POSITIVE.
SPPy itself treats discharge as NEGATIVE current; the sign is flipped inside.

How it works
------------
SPPy's solver is current-driven, but the vehicle model is power-driven. Each
step the pack current is found from  P = (E - R*I) * I  where E is the
"effective source voltage" taken from the PREVIOUS step's terminal voltage
(E = V_prev + R*I_prev), i.e. a semi-implicit scheme. The resulting current is
then used to advance the SPM one step. The SPM's own overpotentials change a
little between steps, so the delivered power differs slightly from the request
(tracked in `max_power_err`).

Pack = one SPM cell scaled to Ns series x Np parallel cells. The default
96s30p of the 1.65 Ah Gao-Randall-Han cell gives about 49.5 Ah and a
voltage curve close to the original ECM pack (about 365 V at 50 % SoC).
That is a lumped model: no cell-to-cell spread, no pack thermal model.
`R_extra` adds pack-level series resistance (busbars, contactors, fuses),
which the single-cell model otherwise lacks.

Temperature (T_batt, degC)
--------------------------
Taken from SPPy's own lumped thermal model, which runs when isothermal=False
(the default here). SPPy updates `cell.T` [K] every step by integrating (rk4)

    rho*Vol*C_p * dT/dt = I*T*(dOCP_p/dT - dOCP_n/dT)      reversible heat
                        + I*(V - (OCP_p - OCP_n))          irreversible heat
                        - h*A*(T - T_amb)                  convection to ambient

and the wrapper reads it back. Things to know:
  * It is a SINGLE-CELL temperature: each cell carries I_pack/Np, with the
    cell's own thermal mass (rho*Vol*C_p) and its own surface (h*A) to ambient.
    Heat sharing between neighbouring cells and pack insulation are not modelled,
    so treat T_batt as a cell temperature, not a pack-core temperature.
  * The irreversible term uses the SPM's own overpotentials (kinetic and
    diffusion), so it is larger than a plain I^2*R estimate.
  * The coupling is two-way: SPPy passes cell.T into its terminal-voltage
    calculation, so temperature feeds back into the voltage the wrapper sees.
  * T_amb defaults to temp_init. Pass T_amb to start the cell away from ambient
    (e.g. temp_init=273.15, T_amb=273.15 for a cold day).
  * With isothermal=True the solver never touches cell.T, so T_batt stays at
    temp_init.
  * h, A, rho, Vol and C_p come from the parameter set (for Gao-Randall-Han:
    h = 1 W/(m^2 K), A = 0.085 m^2), i.e. weak cooling, time constant of
    several minutes. Replace them for a cooled pack.
  * h_cool overrides that h. h_cool=0.0 removes cooling (adiabatic cell): the
    cell then only heats up, and T_amb has no effect on the result (only the
    starting temperature matters).

SPPy setup: clone the repo and put the clone's ROOT folder (the one that
contains both `SPPy/` and `parameter_sets/`) on the path, via the sppy_path
argument, the SPPY_PATH environment variable, or by placing the clone in
this folder (./SPPy) or its parent (../SPPy).
"""
import contextlib
import io
import os
import sys
import warnings

import numpy as np


def _find_sppy_root(explicit=None):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [explicit, os.environ.get("SPPY_PATH"),
             os.path.join(here, "SPPy"), os.path.join(here, "..", "SPPy")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "SPPy", "__init__.py")) \
                and os.path.isdir(os.path.join(c, "parameter_sets")):
            return os.path.abspath(c)
    return None


class SPMBattery:
    def __init__(self, dt=0.1, Ns=96, Np=30, soc0=0.9,
                 param_set="Gao-Randall-Han", isothermal=False, temp_init=298.15,
                 R_extra=0.0, sppy_path=None, T_amb=None, h_cool=None):
        root = _find_sppy_root(sppy_path)
        if root is None:
            raise ImportError(
                "SPPy not found. Clone https://github.com/m0in92/SPPy and either "
                "set SPPY_PATH to the clone's root folder, pass sppy_path=..., or "
                "place the clone at ./SPPy or ../SPPy.")
        if root not in sys.path:
            sys.path.insert(0, root)

        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            import SPPy
            from SPPy.battery_components.parameter_set_manager import ParameterSets
            ps = ParameterSets(name=param_set)
            # pack SoC -> electrode stoichiometries
            sp, sn = SPPy.BatteryCell._get_electrode_soc_from_lib_soc(
                soc0, ps.soc_min_p, ps.soc_max_p, ps.soc_min_n, ps.soc_max_n)
            self.cell = SPPy.BatteryCell.read_from_parametersets(
                param_set, soc_init_p=float(sp), soc_init_n=float(sn),
                temp_init=temp_init)
            self.solver = SPPy.SPPySolver(b_cell=self.cell, isothermal=isothermal,
                                          degradation=False)
            self._invalid_soc_exc = SPPy.warnings_and_exceptions.custom_exceptions \
                .InvalidSOCException
            V0 = self.solver.calc_terminal_potential(I_p_i=0.0, I_n_i=0.0)

        # SPPy sets T_amb equal to the initial temperature; override if asked.
        if T_amb is not None:
            self.cell.T_amb_ = float(T_amb)

        # Heat-transfer coefficient h [W/(m^2 K)]: parameter-set value unless overridden.
        # h_cool=0.0 removes cooling entirely (adiabatic cell).
        if h_cool is not None:
            self.cell.h = float(h_cool)

        self.dt = dt
        self.Ns, self.Np = Ns, Np
        self.R_extra = R_extra
        self.Q = self.cell.cap * Np * 3600.0          # pack capacity [A s]
        self.SoC = soc0
        self.t = 0.0
        self.V_cell = V0                               # rest voltage of one cell
        self.I_pack = 0.0                              # last pack current (+ = discharge)
        self.max_power_err = 0.0                       # worst |P_delivered - P_request| [W]
        self.max_V_cell = V0
        self.min_V_cell = V0
        self.T_batt = self.cell.T - 273.15             # [degC]
        self.max_T_batt = self.T_batt
        self.min_T_batt = self.T_batt

    def _R_pack(self):
        return self.cell.R_cell * self.Ns / self.Np + self.R_extra

    def step(self, P_motor, P_regen):
        net_power = P_motor - P_regen                  # [W], + = discharge
        R = self._R_pack()

        # effective source voltage from the previous step
        V_pack_prev = self.Ns * self.V_cell - self.I_pack * self.R_extra
        E = V_pack_prev + self.I_pack * R
        disc = max(E * E - 4.0 * R * net_power, 0.0)
        I_pack = (E - np.sqrt(disc)) / (2.0 * R)       # smaller-magnitude root
        I_cell = I_pack / self.Np

        try:
            # SPPy convention: charge +, discharge -
            # (with isothermal=False this call also advances cell.T)
            V_cell = self.solver.solve_iteration_one_step(
                t_prev=self.t, dt=self.dt, I=-I_cell)
        except self._invalid_soc_exc as e:
            raise RuntimeError(f"SPM state left its valid range at t={self.t:.1f} s "
                               f"(SoC~{self.SoC:.3f}): {e}") from e
        self.t += self.dt

        V_terminal = self.Ns * V_cell - I_pack * self.R_extra
        self.SoC = min(max(self.SoC - I_pack * self.dt / self.Q, 0.0), 1.0)
        self.V_cell, self.I_pack = V_cell, I_pack
        self.max_V_cell = max(self.max_V_cell, V_cell)
        self.min_V_cell = min(self.min_V_cell, V_cell)
        self.max_power_err = max(self.max_power_err,
                                 abs(V_terminal * I_pack - net_power))

        self.T_batt = self.cell.T - 273.15             # SPPy cell temperature [degC]
        self.max_T_batt = max(self.max_T_batt, self.T_batt)
        self.min_T_batt = min(self.min_T_batt, self.T_batt)
        return dict(I_batt=I_pack, net_power=net_power,
                    V_terminal=V_terminal, SoC=self.SoC, T_batt=self.T_batt)