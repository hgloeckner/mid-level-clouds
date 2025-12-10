# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from moist_thermodynamics import constants
import sys
import xarray as xr
from xhistogram.xarray import histogram

sys.path.append("../")
import myutils.open_datasets as open_datasets
from myutils.physics_helper import (
    get_stability,
    density_from_q,
    get_csc_stab,
    get_csc_cooling,
)
import myutils.data_helper as dh

# %%
flux_data_arts2 = open_datasets.open_radiative_fluxes()

cid = "ipfs://bafybeiesyutuduzqwvu4ydn7ktihjljicywxeth6wtgd5zi4ynxzqngx4m"
lvl3 = open_datasets.open_dropsondes(cid)

lvl3 = (
    lvl3.where((lvl3.p_qc == 0) & (lvl3.ta_qc == 0) & (lvl3.rh_qc == 0), drop=True)
    .pipe(dh.interpolate_gaps)
    .pipe(dh.extrapolate_sfc)
    .swap_dims({"sonde": "sonde_id"})
)
lvl3 = lvl3  # .where(lvl3.sonde_id.isin(cs_sondes), drop=True)

# %%
stability = (
    get_stability(
        lvl3.theta,
        lvl3.ta,
    )
    .coarsen(altitude=10)
    .mean()
    .rolling(altitude=10)
    .mean()
)
stability["altitude"] = stability["altitude"] + 5
stability.name = "stability"

rho = (
    density_from_q(
        lvl3.p,
        lvl3.ta,
        lvl3.q,
    )
    .coarsen(altitude=10)
    .mean()
    .rolling(altitude=10)
    .mean()
)

rho["altitude"] = rho["altitude"] + 5
rho.name = "rho"

t = (
    lvl3.ta.interpolate_na("altitude", fill_value="extrapolate")
    .coarsen(altitude=10)
    .mean()
    .rolling(altitude=10)
    .mean()
)
t["altitude"] = t["altitude"] + 5
# %%#
# %%
H = flux_data_arts2.cooling_rate * constants.cpv * rho  # J m-3 day-1
# %%
csc_stab = get_csc_stab(rho, stability, H).compute()
csc_stab.name = "csc_stab"
csc_cooling = get_csc_cooling(rho, stability, H).compute()
csc_cooling.name = "csc_cooling"

# %%


histds = flux_data_arts2.sel(altitude=slice(0, 15000))
cr_bins = np.linspace(0, 5, 50)
ta_bins = np.linspace(210, 305, 100)
stab_bins = np.linspace(0, 8, 100)
csc_bins = np.linspace(-5, 5, 100)
# %%
ta_hists = {}

for ta, vards, bins in [
    (histds.ta, histds.cooling_rate, cr_bins),
    (t, stability, stab_bins),
    (
        t.isel(altitude=slice(0, -1)).where(
            t.sonde_id.isin(csc_stab.sonde_id), drop=True
        ),
        csc_stab,
        csc_bins,
    ),
    (
        t.isel(altitude=slice(0, -1)).where(
            t.sonde_id.isin(csc_cooling.sonde_id), drop=True
        ),
        csc_cooling,
        csc_bins,
    ),
]:
    hist = histogram(
        ta,
        vards,
        dim=["altitude"],
        bins=[ta_bins, bins],
    ).compute()
    hist = hist.assign_coords(
        launch_lat=csc_stab.launch_lat,
        launch_lon=csc_stab.launch_lon,
    )
    ta_hists[f"{vards.name}_of_ta"] = hist


# %%

# %%
fig, axes = plt.subplots(ncols=3, sharey=True, sharex=True, figsize=(15, 5))


for region, ax in zip([dh.east, dh.west, dh.north], axes):
    for name in ["csc_cooling", "csc_stab"]:
        pltval = (
            dh.sel_sub_domain(ta_hists[f"{name}_of_ta"], region, item_var="sonde_id")
            .mean("sonde_id")
            .rolling(ta_bin=5)
            .mean()
        )
        pltval = (pltval * pltval[f"{name}_bin"]).sum(f"{name}_bin") / pltval.sum(
            f"{name}_bin"
        )

        pltval.rolling(ta_bin=5).mean().plot(
            y="ta_bin",
            ax=ax,
        )
    pltval = (
        dh.sel_sub_domain(
            (
                ta_hists["csc_cooling_of_ta"].rename(
                    {"csc_cooling_bin": "csc_stab_bin"}
                )
                + ta_hists["csc_stab_of_ta"]
            ),
            region,
            item_var="sonde_id",
        )
        .mean("sonde_id")
        .rolling(ta_bin=5)
        .mean()
    )
    pltval = (pltval * pltval["csc_stab_bin"]).sum("csc_stab_bin") / pltval.sum(
        "csc_stab_bin"
    )

    pltval.rolling(ta_bin=5).mean().plot(
        y="ta_bin",
        color="k",
        ax=ax,
    )

axes[0].invert_yaxis()
axes[0].set_xlim(None, 0.5)
for ax in axes:
    ax.axvline(0, color="grey", linestyle=":")
    ax.axhline(273.15, color="grey", linestyle=":", alpha=0.2)

for ax, yval in zip(axes, [269, 272, 265]):
    ax.axhline(yval, color="grey", linestyle="-")
sns.despine()

# %%
lvl4 = xr.open_dataset(
    "ipfs://bafybeihfqxfckruepjhrkafaz6xg5a4sepx6ahhv4zds4b3hnfiyj35c5i", engine="zarr"
)
regions = [dh.east, dh.west, dh.north]

for region in regions:
    pltds = dh.sel_sub_domain(
        lvl4, region, item_var="circle", lon_var="circle_lon", lat_var="circle_lat"
    )
    plt.plot(pltds.div.mean("circle"), pltds.ta_mean.mean("circle"))
# %%
fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(10, 5))

name = "stability"
for region, label in zip([dh.east, dh.west, dh.north], ["East", "West", "North"]):
    for name, ax in zip(["stability", "cooling_rate"], axes):
        pltval = (
            dh.sel_sub_domain(ta_hists[f"{name}_of_ta"], region, item_var="sonde_id")
            .mean("sonde_id")
            .rolling(ta_bin=5)
            .mean()
        )

        (
            (pltval * pltval[f"{name}_bin"]).sum(f"{name}_bin")
            / pltval.sum(f"{name}_bin")
        ).plot(
            y="ta_bin",
            ax=ax,
            label=label,
        )
        ax.set_ylabel("")
    """
    pltval = dh.sel_sub_domain((ta_hists["csc_cooling_of_ta"].rename({"csc_cooling_bin":"csc_stab_bin"}) + ta_hists["csc_stab_of_ta"]), region, item_var="sonde_id").mean("sonde_id").rolling(ta_bin=5).mean()
    pltval = (pltval * pltval[f"csc_stab_bin"]).sum(f"csc_stab_bin") / pltval.sum(f"csc_stab_bin")
    pltval.plot(
        ax=axes[2],
        y="ta_bin",
    )
    """
    pltds = (
        dh.sel_sub_domain(
            lvl4, region, item_var="circle", lon_var="circle_lon", lat_var="circle_lat"
        )
        .rolling(altitude=5)
        .mean()
    )
    axes[2].plot(
        pltds.div.where(
            pltds.omega.sel(altitude=slice(5000, 10000)).mean("altitude") < 0
        ).mean("circle"),
        pltds.ta_mean.where(
            pltds.omega.sel(altitude=slice(5000, 10000)).mean("altitude") < 0
        ).mean("circle"),
    )
axes[0].legend()
axes[0].invert_yaxis()
axes[0].set_ylim(295, 250)
axes[0].set_xlim(2, 6)
axes[1].set_xlim(1.5, None)
axes[0].set_xlabel("stability / K m$^{-1}$")
axes[0].set_ylabel("temperature / K")
axes[1].set_xlabel("cooling rate / K day$^{-1}$")
# axes[2].set_xlim(-0.2, 0.2)
axes[2].set_ylabel("")
# axes[2].set_xlabel("CSC (Spaulding)")
axes[2].axvline(0, color="grey")
axes[2].axhline(273.15, color="grey", linestyle=":", alpha=0.2)

sns.despine(offset=10)
