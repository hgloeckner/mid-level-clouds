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
from myutils.data_helper import get_hist_of_ta

# %%
cid = opends.get_cid()
rs = opends.open_radiosondes(f"{cid}/Radiosondes/RAPSODI_RS_ORCESTRA_level2.zarr")


ds = opends.open_dropsondes(f"{cid}/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr")

gate = opends.open_gate("QmeAFUdB3PZHRtCd441HjRGZPmEadtskXsnL34C9xigH3A")

unique, keep = np.unique(gate.sonde_id.values, return_index=True)
gate = gate.isel(sonde_id=keep)
# %%

# %%

lats = [5, 12]
lons = [-27, -20]
orcestra_gate = xr.concat(
    [
        rs.where(
            (lons[0] < rs.launch_lon)
            & (rs.launch_lon < lons[1])
            & (lats[0] < rs.launch_lat)
            & (rs.launch_lat < lats[1])
            & (rs.ascent_flag == 0),
            drop=True,
        ),
        ds.where(
            (lons[0] < ds.launch_lon)
            & (ds.launch_lon < lons[1])
            & (lats[0] < ds.launch_lat)
            & (ds.launch_lat < lats[1]),
            drop=True,
        ),
    ],
    dim="sonde_id",
)
rs_gate = orcestra_gate.where(orcestra_gate.sonde_id.isin(rs.sonde_id), drop=True)
ds_gate = orcestra_gate.where(orcestra_gate.sonde_id.isin(ds.sonde_id), drop=True)
gate_region = gate.where(
    (lons[0] < gate.launch_lon)
    & (gate.launch_lon < lons[1])
    & (lats[0] < gate.launch_lat)
    & (gate.launch_lat < lats[1]),
    drop=True,
)
# %%
es_liq = svp.liq_analytic
es_ice = svp.ice_analytic
es = mt.make_es_mxd(es_liq=svp.liq_analytic, es_ice=svp.ice_analytic)
Rd = constants.Rd
Rv = constants.Rv
cpd = constants.cpd
P0 = constants.P0


def get_rh(T, q, p, es):
    x = es(T) * Rd / Rv / (p - es(T))
    return q * (1 + x) / x


def get_q(T, rh, p, es):
    x = es(T) * Rd / Rv / (p - es(T))
    return rh * x / (1 + x)


q = get_q(195, 1, 150, es=es_ice)
rh_5 = get_rh(200, q, 150, es=es_ice)
# %%

gate_region = gate_region.assign(
    rh_ice=get_rh(gate_region.ta, gate_region.q, gate_region.p, es=svp.ice_analytic),
    rh_liq=get_rh(gate_region.ta, gate_region.q, gate_region.p, es=svp.liq_analytic),
)
rs_gate = rs_gate.assign(
    rh_ice=get_rh(rs_gate.ta, rs_gate.q, rs_gate.p, es=svp.ice_analytic),
    rh_liq=get_rh(rs_gate.ta, rs_gate.q, rs_gate.p, es=svp.liq_analytic),
)
ds_gate = ds_gate.assign(
    rh_ice=get_rh(ds_gate.ta, ds_gate.q, ds_gate.p, es=svp.ice_analytic),
    rh_liq=get_rh(ds_gate.ta, ds_gate.q, ds_gate.p, es=svp.liq_analytic),
)

# %%
print(
    "Rapsodi",
    rs_gate.ta.min(dim="altitude").mean().values,
    "Gate",
    gate_region.ta.where(gate_region.ta.count(dim="altitude") > 1900)
    .min(dim="altitude")
    .mean()
    .values,
)

# %%
gate_strato = gate_region.where(
    (gate_region.altitude > gate_region.ta.argmin(dim="altitude") * 10)
    & (gate_region.ta.count(dim="altitude") > 1900)
)
rs_strato = rs_gate.where(rs_gate.altitude > rs_gate.ta.argmin(dim="altitude") * 10)
# %%
rs_strato = rs_strato.assign(diff_to_cp=rs_strato.ta - rs_strato.ta.min(dim="altitude"))
gate_strato = gate_strato.assign(
    diff_to_cp=gate_strato.ta - gate_strato.ta.min(dim="altitude")
)

# %%
thres = 10

sns.histplot(
    rs_strato.rh_liq.where(
        (rs_strato.diff_to_cp > thres) & (rs_strato.diff_to_cp < thres + 0.5)
    ).mean(dim="altitude"),
    bins=30,
    stat="density",
    label="RAPSODI",
    kde=True,
    element="step",
)
sns.histplot(
    gate_strato.rh_liq.where(
        (gate_strato.diff_to_cp > thres) & (gate_strato.diff_to_cp < thres + 0.5)
    ).mean(dim="altitude"),
    bins=30,
    stat="density",
    label="GATE",
    kde=True,
    element="step",
)

# %%
distribution_altitude = [2000, 8000]
cs_threshold = 1

fig, axes = plt.subplots(
    ncols=len(distribution_altitude),
    figsize=((len(distribution_altitude)) * 5, 5),
)
for ax, alt in zip(axes, distribution_altitude):
    sns.histplot(
        rs_gate.rh.sel(altitude=alt),
        bins=30,
        stat="density",
        label="RAPSODI",
        kde=True,
        element="step",
        ax=ax,
    )
    sns.histplot(
        gate_region.where(gate_region.rh.max(dim="altitude") < cs_threshold).rh.sel(
            altitude=alt
        ),
        bins=30,
        stat="density",
        label="GATE",
        kde=True,
        element="step",
        ax=ax,
    )
    ax.set_title(f"Altitude: {alt} m")

for ax in axes:
    ax.set_xlabel("Relative Humidity")
    ax.legend()
    ax.set_xlim(0, 1.1)
sns.despine()
# %%
rapsodi_ta = get_hist_of_ta(
    rs_gate.ta.sel(altitude=slice(0, 14000)),
    rs_gate.rh.sel(altitude=slice(0, 14000)),
    bins_var=np.linspace(0, 1.1, 100),
    bins_ta=np.linspace(220, 305, 200),
)
beach_ta = get_hist_of_ta(
    ds_gate.ta.sel(altitude=slice(0, 14000)),
    ds_gate.rh.sel(altitude=slice(0, 14000)),
    bins_var=np.linspace(0, 1.1, 100),
    bins_ta=np.linspace(220, 305, 200),
)

gate_ta = get_hist_of_ta(
    gate_region.ta.sel(altitude=slice(0, 14000)),
    gate_region.rh.sel(altitude=slice(0, 14000)),
    bins_var=np.linspace(0, 1.1, 100),
    bins_ta=np.linspace(220, 305, 200),
)
rapsodi_ta_ice = get_hist_of_ta(
    rs_gate.ta.sel(altitude=slice(0, 14000)),
    rs_gate.rh_ice.sel(altitude=slice(0, 14000)),
    bins_var=np.linspace(0, 1.1, 100),
    bins_ta=np.linspace(220, 305, 200),
)
beach_ta_ice = get_hist_of_ta(
    ds_gate.ta.sel(altitude=slice(0, 14000)),
    ds_gate.rh_ice.sel(altitude=slice(0, 14000)),
    bins_var=np.linspace(0, 1.1, 100),
    bins_ta=np.linspace(220, 305, 200),
)

gate_ta_ice = get_hist_of_ta(
    gate_region.ta.sel(altitude=slice(0, 14000)),
    gate_region.rh_ice.sel(altitude=slice(0, 14000)),
    bins_var=np.linspace(0, 1.1, 100),
    bins_ta=np.linspace(220, 305, 200),
)


# %%
distribution_t = [280, 255]
thres = 5
fig, axes = plt.subplots(
    ncols=len(distribution_t) + 1, figsize=((len(distribution_t) + 1) * 5, 5)
)
ax = axes[-1]
for ax, t in zip(axes[:-1], distribution_t):
    if t > 273.15:
        rs_ds = rapsodi_ta
        gate_ds = gate_ta
        subscript = "liq"
    else:
        rs_ds = rapsodi_ta_ice
        gate_ds = gate_ta_ice
        subscript = "ice"
    sns.histplot(
        rs_ds.sel(ta=t, method="nearest"),
        bins=30,
        stat="density",
        label=r"RAPSODI (RH$_{{{}}}$)".format(subscript),
        kde=True,
        element="step",
        ax=ax,
    )
    sns.histplot(
        gate_ds.sel(ta=t, method="nearest"),
        bins=30,
        stat="density",
        label=r"GATE (RH$_{{{}}}$)".format(subscript),
        kde=True,
        element="step",
        ax=ax,
    )
    """
    sns.histplot(
        beach_ta.sel(ta=t, method="nearest"),
        bins=30,
        stat="density",
        label="BEACH",
        kde=True,
        element="step",
        ax=ax,
    )
    """
    ax.set_title(f"troposphere temperature: {t} K")

ax = axes[-1]
sns.histplot(
    rs_strato.rh_ice.where(
        (rs_strato.diff_to_cp > thres) & (rs_strato.diff_to_cp < thres + 0.5)
    ).mean(dim="altitude"),
    bins=30,
    stat="density",
    label=r"RAPSODI (RH$_{{ice}}$)",
    kde=True,
    element="step",
    ax=ax,
)
sns.histplot(
    gate_strato.rh_ice.where(
        (gate_strato.diff_to_cp > thres) & (gate_strato.diff_to_cp < thres + 0.5)
    ).mean(dim="altitude"),
    bins=30,
    stat="density",
    label=r"GATE (RH$_{{ice}}$)",
    kde=True,
    element="step",
    ax=ax,
)

ax.set_title(f"stratosphere: {thres} K < $T$ - $T_{{min}}$ < {thres + 0.5} K")


for ax in axes:
    ax.legend()
    ax.set_xlabel("Relative Humidity")
    ax.set_xlim(0, 1.1)
sns.despine()
