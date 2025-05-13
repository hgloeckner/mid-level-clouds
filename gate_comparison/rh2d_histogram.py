# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from xhistogram.xarray import histogram
import moist_thermodynamics.functions as mt
from moist_thermodynamics import saturation_vapor_pressures as svp

from moist_thermodynamics import constants
import sys

sys.path.append("../")
import myutils.open_datasets as opends  # noqa
import myutils.plot_helper as ph  # noqa

# %%
es = mt.make_es_mxd(es_liq=svp.liq_analytic, es_ice=svp.ice_analytic)

# %%
rs = opends.open_radiosondes(
    "bafybeigensqyqxfyaxgyjhwn6ytdpi3i4sxbtffd4oc27zbimyro4hygjq"
)
rs = rs.where(rs.launch_lon > -40, drop=True)


ds = opends.open_dropsondes(
    "bafybeicb33v6ohezyhgq5rumq4g7ejnfqxzcpuzd4i2fxnlos5d7ebmi3m"
)
ds = ds.where(ds.launch_lon > -40, drop=True)

gate = opends.open_gate("QmWmYbYbW51bpYGREctj1LLWSMrPc7sEXkgDzhsDYsW3qg")

unique, keep = np.unique(gate.sonde_id.values, return_index=True)
gate = gate.isel(sonde_id=keep)
gate = gate.where(gate.launch_lon > -40, drop=True)

# %%

# %%
fig, ax = plt.subplots(figsize=(6, 6))

for data, label in zip([rs, ds, gate], ["rapsody", "beach", "gate"]):
    sns.histplot(
        data=data.rh.sel(altitude=slice(0, 14000)).to_dataframe(),
        x="rh",
        ax=ax,
        bins=100,
        binrange=(0, 1.1),
        stat="percent",
        element="step",
        alpha=0.5,
        label=label,
    )

sns.despine(offset={"left": 10})
ax.set_xlim(0, 1.1)
ax.legend()

# %%
es_liq = svp.liq_analytic
es_ice = svp.liq_analytic
es = mt.make_es_mxd(es_liq=svp.liq_analytic, es_ice=svp.ice_analytic)
Rd = constants.Rd
Rv = constants.Rv
cpd = constants.cpd
P0 = constants.P0


ta = np.linspace(220, 305, 100)


def get_ice_liq(ta, theta, q):
    ds_p = (ta / theta) ** (cpd / Rd) * P0
    R = Rd + (Rv - Rd) * q
    pv = q * Rv / R * ds_p
    return pv / es(ta)


def get_rh_from_q(ta, p):
    return es(ta) * Rd / (Rv * (p - es(ta)))


def get_rh_ice_from_liq(ta, da):
    p = (
        da.mean("sonde_id")
        .sel(altitude=slice(0, 14000))
        .interpolate_na(dim="altitude", fill_value="extrapolate")
        .swap_dims({"altitude": "ta"})
        .p.interp(ta=ta, kwargs={"fill_value": "extrapolate"})
    )
    vmr100 = es_ice(ta) / p
    return vmr100 * p / es_liq(ta)


p_beach = get_rh_ice_from_liq(ta, ds)
p_gate = get_rh_ice_from_liq(ta, gate)
p_rap = get_rh_ice_from_liq(ta, rs)
#
# %%

orcestra_total = xr.concat(
    [
        ds,
        rs,
    ],
    dim="sonde_id",
    data_vars="minimal",
)


fig, axes = plt.subplots(ncols=2, figsize=(15, 6))
for data, ax, label, rh_ice in zip(
    [orcestra_total.where(orcestra_total.launch_lon > -40, drop=True), gate],
    axes,
    ["orcestra (east)", "gate"],
    [p_rap, p_beach, p_gate],
):
    hist = histogram(
        data.ta.sel(altitude=slice(0, 14000)),
        data.rh.sel(altitude=slice(0, 14000)),
        bins=[np.linspace(220, 305, 200), np.linspace(0, 1.1, 200)],
    )
    hist = hist / hist.sum(dim=["ta_bin", "rh_bin"])
    p = hist.plot(
        ax=ax,
        cmap="Blues",
        vmax=0.00008,
        add_colorbar=False,
    )

    ax.invert_yaxis()
    ax.set_title(label)
    ax.axhline(273.15, color="k", linestyle="--")
for ax in axes:
    ax.plot(
        rh_ice,
        ta,
        color="black",
    )
ph.plot_cbar(fig, p, ax, "count / total", "1")
sns.despine(offset={"bottom": 10})
