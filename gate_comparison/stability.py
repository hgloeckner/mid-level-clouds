# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import moist_thermodynamics.functions as mt
from moist_thermodynamics import saturation_vapor_pressures as svp

from moist_thermodynamics import constants

import sys

sys.path.append("../")
import myutils.open_datasets as opends  # noqa

es = mt.make_es_mxd(es_liq=svp.liq_analytic, es_ice=svp.ice_analytic)
# %%
radios = opends.open_radiosondes(
    "bafybeigensqyqxfyaxgyjhwn6ytdpi3i4sxbtffd4oc27zbimyro4hygjq"
)

drops = opends.open_dropsondes(
    "bafybeicb33v6ohezyhgq5rumq4g7ejnfqxzcpuzd4i2fxnlos5d7ebmi3m"
)
gate = opends.open_gate("QmWmYbYbW51bpYGREctj1LLWSMrPc7sEXkgDzhsDYsW3qg")

unique, keep = np.unique(gate.sonde_id.values, return_index=True)
gate = gate.isel(sonde_id=keep)
datasets = {
    "radiosonde": radios,
    "dropsonde": drops,
    "gate": gate,
}
# %%


def get_n2(th, qv, altdim="altitude"):
    """Returns the Brunt-Vaisala frequeny for unsaturated air.

    It assumes that the input are type xarray with their first coordinate being
    altitude in meters, and that the air is no where saturated

    Args:
        th: potential temperature
        qv: specific humidity
    """

    from moist_thermodynamics import constants

    Rv = constants.water_vapor_gas_constant
    Rd = constants.dry_air_gas_constant
    g = constants.gravity_earth
    R = Rd + (Rv - Rd) * qv
    dlnthdz = np.log(th).differentiate(altdim)
    dqvdz = qv.differentiate(altdim)

    return np.sqrt(g * (dlnthdz + (Rv - Rd) * dqvdz / R))


# %%

es_liq = svp.liq_analytic
es_ice = svp.liq_analytic
es = mt.make_es_mxd(es_liq=svp.liq_analytic, es_ice=svp.ice_analytic)
Rd = constants.Rd
Rv = constants.Rv
cpd = constants.cpd
P0 = constants.P0


def get_ice_liq(ta, theta, q):
    ds_p = (ta / theta) ** (cpd / Rd) * P0
    R = Rd + (Rv - Rd) * q
    pv = q * Rv / R * ds_p
    return pv / es(ta)


# %%

# %%
sns.set_palette("Paired")
fig, ax = plt.subplots(1, 1, figsize=(4, 5), sharey=True)
# plot radiosondes
for dataname, data in datasets.items():
    rh = get_ice_liq(data.ta, data.theta, data.q)
    rh.mean("sonde_id").plot(
        y="altitude", xlim=(0, 1), ylim=(0, 23000), label=f"ice-liq {dataname}"
    )
    data.rh.mean("sonde_id").plot(
        y="altitude", xlim=(0, 1), ylim=(0, 23000), label="liq (rad)"
    )

ax.set_ylabel("altitude / m")
ax.set_xlabel("RH")
sns.despine(offset=10)
plt.legend()
# %%

# rs = rs.where((rs.rh < 1) & (rs.rh >0))
for dataname, data in datasets.items():
    datasets[dataname] = data.assign(
        stability=get_n2(
            th=data.theta.interpolate_na("altitude"),
            qv=data.q.interpolate_na("altitude"),
        )
    )

# %%
data = datasets["radiosonde"]
mean_t = data.ta.mean("sonde_id").sel(altitude=slice(0, 14000))
mean_q = data.q.mean("sonde_id").sel(altitude=slice(0, 14000))
mean_p = data.p.mean("sonde_id").sel(altitude=slice(0, 14000))

theta_e_bolton = mt.theta_e_bolton(
    T=mean_t,
    P=mean_p,
    qt=mean_q,
    es=svp.liq_analytic,
)
# %%
temp = []
for alt in mean_p.altitude.values:
    temp.append(
        mt.invert_for_temperature(
            f=mt.theta_e_bolton,
            f_val=theta_e_bolton.sel(altitude=0).values,
            P=mean_p.sel(altitude=alt).values,
            qt=mean_q.sel(altitude=0).values,
            es=svp.liq_analytic,
        )
    )

theta = mt.theta(T=temp, P=mean_p, qv=mean_q)
stab = get_n2(theta, mean_q, altdim="altitude")
# %%
sns.set_palette("colorblind")
fig, ax = plt.subplots(figsize=(6, 5), sharey=True)

for dataname, data in datasets.items():
    """
    data.stability.mean("sonde_id").plot(
        label=dataname,
        y="altitude",
    )
    """
    ax.plot(
        data.stability.mean("sonde_id").sel(altitude=slice(0, 14000)).values,
        data.ta.mean("sonde_id").sel(altitude=slice(0, 14000)).values,
        label=dataname,
    )
ax.plot(
    stab.rolling(altitude=15).mean(),
    temp,
    label="pseudo",
    color="black",
)

ax.legend()
ax.invert_yaxis()
ax.set_ylim(300, 220)
ax.set_xlim(0, 0.02)
ax.axhline(273.15, color="k", linestyle="-", lw=0.5)
sns.despine(offset={"bottom": 10})
# %%

# %%
