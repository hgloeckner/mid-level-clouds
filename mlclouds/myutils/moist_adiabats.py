# %%

import numpy as np
import moist_thermodynamics.constants as mtc
import moist_thermodynamics.utilities as mtu
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp
import xarray as xr
from scipy.integrate import solve_ivp


es = svp.liq_wagner_pruss


def qs(P, T):
    e = es(T)
    return mtc.eps1 * e / (P - e)


def pseudo_adiabat(T0, P, Tmin, qt):
    def dT_dP(p, T):
        qsat = qs(p, T)

        cp = mtc.cpd + qsat * mtc.cpv
        R = mtc.Rd + qsat * mtc.Rv

        num = (R * T) / (p * cp) * (1 + (mtc.lv0 * qsat) / (mtc.Rd * T))
        den = 1 + (mtc.lv0**2 * qsat) / (cp * mtc.Rv * T**2)

        return num / den

    sol = solve_ivp(dT_dP, [P[0], P[-1]], [T0], t_eval=P)
    return np.maximum(sol.y[0], Tmin)


def reversible_adiabat(T0, P, Tmin, qt):
    def dT_dP(p, T):
        qsat = qs(p, T)

        qv = min(qsat, qt)
        qc = qt - qv
        qd = 1 - qt

        cp = qd * mtc.cpd + qv * mtc.cpv + qc * mtc.cl
        R = qd * mtc.Rd + qv * mtc.Rv

        alpha = R * T / p

        dX_dT = cp
        dX_dP = alpha

        if qc > 0:
            beta_p = R / (qd * mtc.Rd)
            beta_t = beta_p * mtc.lv0 / (mtc.Rv * T)

            dX_dT += mtc.lv0 * qv * beta_t / T
            dX_dP *= 1 + mtc.lv0 * qv * beta_p / (R * T)

        return dX_dP / dX_dT

    sol = solve_ivp(dT_dP, [P[0], P[-1]], [T0], t_eval=P)
    return np.maximum(sol.y[0], Tmin)


def make_sounding_from_adiabat(
    ma_fct,
    P,
    Tsfc=301.0,
    Tmin=200.0,
    qsfc=17e-3,
) -> xr.Dataset:
    """creates a sounding from a moist adiabat

    Cacluates the moist adiabate based either on an integration or a specified
    isentrope with pressure as the vertical coordinate.

    Args:
        P: pressure
        Tsfc: starting (value at P.max()) temperature
        qsfc: starting (value at P.max()) specific humidity
        Tmin: minimum temperature of adiabat
        thx: function to calculate isentrope if integrate = False
        integrate: determines if explicit integration will be used.
    """

    TPq = xr.Dataset(
        data_vars={
            "T": (
                ("levels",),
                ma_fct(P=P, T0=Tsfc, Tmin=Tmin, qt=qsfc),
                {"units": "K", "standard_name": "air_temperature", "symbol": "$T$"},
            ),
            "P": (
                ("levels",),
                P,
                {"units": "Pa", "standard_name": "air_pressure", "symbol": "$P$"},
            ),
            "q": (
                ("levels",),
                qsfc * np.ones(len(P)),
                {"units": "1", "standard_name": "specific_humidity", "symbol": "$q$"},
            ),
        },
    )
    TPq = TPq.assign(
        altitude=xr.DataArray(
            mtf.pressure_altitude(TPq.P, TPq.T, qv=TPq.q).values,
            dims=("levels"),
            attrs={
                "units": "m",
                "standard_name": "altitude",
                "description": "hydrostatic altitude given the datasets temperature and pressure",
            },
        )
    )
    TPq = TPq.assign(
        theta=(
            TPq.T.dims,
            mtf.theta(TPq.T, TPq.P).values,
            {
                "units": "K",
                "standard_name": "air_potential_teimerature",
                "symbol": "$\theta$",
            },
        )
    )
    TPq = TPq.assign(
        P0=xr.DataArray(
            mtc.P0, attrs={"units": "Pa", "standards_name": "referenece_pressure"}
        )
    )

    return TPq.set_coords("altitude").swap_dims({"levels": "altitude"})


# %%
"""
Psfc = 101021
qsfc = 18e-3
Tsfc = 301.4

P = np.arange(Psfc, 4000.0, -500)

pseudo = make_sounding_from_adiabat(pseudo_adiabat, P, Tsfc=Tsfc, Tmin=200.0, qsfc=qsfc)
reversible = make_sounding_from_adiabat(reversible_moist_adiabat, P, Tsfc=Tsfc, Tmin=200.0, qsfc=qsfc)

"""
