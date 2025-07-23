# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import moist_thermodynamics.functions as mt
from moist_thermodynamics import saturation_vapor_pressures as svp

from moist_thermodynamics import constants
import sys

sys.path.append("../")
import myutils.open_datasets as opends  # noqa
import myutils.plot_helper as ph  # noqa
from myutils.data_helper import get_hist_of_ta, get_hist_of_ta_2d, get_gate_region

# %%
cid = opends.get_cid()
rs = opends.open_radiosondes(f"{cid}/Radiosondes/RAPSODI_RS_ORCESTRA_level2.zarr")


ds = opends.open_dropsondes(f"{cid}/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr")

gate = opends.open_gate("QmeAFUdB3PZHRtCd441HjRGZPmEadtskXsnL34C9xigH3A")

unique, keep = np.unique(gate.sonde_id.values, return_index=True)
gate = gate.isel(sonde_id=keep)
# %%

orcestra_gate, gate_region = get_gate_region(
    gate, rs=rs, ds=ds, ascent_flag=0, lats=(5, 12), lons=(-27, -20)
)

# %%
orcestra_ta = get_hist_of_ta(
    orcestra_gate.ta.sel(altitude=slice(0, 14000)),
    orcestra_gate.rh.sel(altitude=slice(0, 14000)),
    bins_var=np.linspace(0, 1.1, 100),
    bins_ta=np.linspace(220, 305, 200),
)
gate_ta = get_hist_of_ta(
    gate_region.ta.sel(altitude=slice(0, 14000)),
    gate_region.rh.sel(altitude=slice(0, 14000)),
    bins_var=np.linspace(0, 1.1, 100),
    bins_ta=np.linspace(220, 305, 200),
)
# %%

orcestra_2dhist = get_hist_of_ta_2d(
    orcestra_gate.ta.sel(altitude=slice(0, 14000)),
    orcestra_gate.rh.sel(altitude=slice(0, 14000)),
    bins_var=np.linspace(0, 1.1, 100),
    bins_ta=np.linspace(220, 305, 200),
)
gate_2dhist = get_hist_of_ta_2d(
    gate_region.ta.sel(altitude=slice(0, 14000)),
    gate_region.rh.sel(altitude=slice(0, 14000)),
    bins_var=np.linspace(0, 1.1, 100),
    bins_ta=np.linspace(220, 305, 200),
)
# %%
(orcestra_2dhist / orcestra_2dhist.sum("rh_bin")).plot(
    vmax=0.05, cmap="Blues", alpha=0.5
)
(gate_2dhist / gate_2dhist.sum("rh_bin")).plot(vmax=0.05, cmap="Reds", alpha=0.5)
# %%
cs_threshold = 0.95
fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
ax = axes[0]
orcestra_ta.mean("sonde_id").rolling(ta=5).mean().plot(
    label="ORCESTRA", y="ta", color="blue", ax=ax
)
gate_ta.mean("sonde_id").rolling(ta=5).mean().plot(
    label="GATE", y="ta", color="red", ax=ax
)


(orcestra_2dhist / orcestra_2dhist.sum("rh_bin")).plot(
    vmax=0.05,
    cmap="Blues",
    ax=ax,
    y="ta_bin",
    x="rh_bin",
    add_colorbar=False,
)
ax = axes[1]
orcestra_ta.mean("sonde_id").rolling(ta=5).mean().plot(
    label="ORCESTRA", y="ta", color="blue", ax=ax
)
gate_ta.mean("sonde_id").rolling(ta=5).mean().plot(
    label="GATE", y="ta", color="red", ax=ax
)


(gate_2dhist / gate_2dhist.sum("rh_bin")).plot(
    vmax=0.05,
    cmap="Reds",
    ax=ax,
    y="ta_bin",
    x="rh_bin",
    add_colorbar=False,
)


ax.legend()
for ax in axes:
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.set_xlabel("Relative humidity")
    ax.axhline(273.15, color="k", linestyle="--", linewidth=0.5)

axes[0].set_ylabel("Temperature / K")
fig.suptitle(
    "RH histograms for ORCESTRA and GATE and mean; lons in [-27, -20], lats in [5, 12]"
)
sns.despine(offset={"bottom": 10})
fig.savefig(
    "../plots/gate/rh_histograms.pdf",
)
# %%
colors = sns.color_palette("Paired", n_colors=8)
cs_threshold = 0.95
fig, ax = plt.subplots(figsize=(5, 5))

gate_ta.where(gate_ta.max(dim="ta") < cs_threshold).mean("sonde_id").rolling(
    ta=5
).mean().plot(y="ta", ax=ax, label=f"GATE rh_max < {cs_threshold:.2f}", c=colors[4])
gate_ta.mean("sonde_id").rolling(ta=5).mean().plot(
    label="GATE", y="ta", ax=ax, c=colors[5]
)
orcestra_ta.where(orcestra_ta.max(dim="ta") < cs_threshold).mean("sonde_id").rolling(
    ta=5
).mean().plot(y="ta", ax=ax, label=f"ORCESTRA rh_max < {cs_threshold:.2f}", c=colors[0])
orcestra_ta.mean("sonde_id").rolling(ta=5).mean().plot(
    label="ORCESTRA", y="ta", ax=ax, c=colors[1]
)


ax.invert_yaxis()
ax.legend()
ax.set_xlabel("Relative humidity")
ax.set_ylabel("Temperature / K")
sns.despine(offset=10)
fig.savefig(
    "../plots/gate/total_vs_clear_sky.pdf",
)
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
es_ice = svp.ice_analytic
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


def get_rh(T, q, p, es):
    x = es(T) * Rd / Rv / (p - es(T))
    return q * (1 + x) / x


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
