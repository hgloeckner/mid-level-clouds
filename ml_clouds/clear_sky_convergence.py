# %%
from xhistogram.xarray import histogram
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from moist_thermodynamics import constants
import sys

sys.path.append("../")
from myutils.constants_and_values import ml_sondes, cs_sondes
import myutils.open_datasets as open_datasets

# %%
flux_data_arts2 = open_datasets.open_radiative_fluxes()
IPFS_GATEWAY = "https://ipfs.io"
cid = open_datasets.get_cid()
lvl3 = open_datasets.open_dropsondes(f"{cid}/dropsondes/Level_3/PERCUSION_Level_3.zarr")

lvl3 = lvl3.where((lvl3.p_qc == 0) & (lvl3.ta_qc == 0) & (lvl3.rh_qc == 0), drop=True)


# %%
def get_stability(theta, T):
    return (T / theta * theta.differentiate("altitude")) * 1000


def get_csc_stab(rho, stability, H):
    grad_stability = stability.differentiate("altitude") * 1000
    cp = constants.cpv
    return 1 / (cp * rho * stability) * (H / stability * grad_stability)


def get_csc_cooling(rho, stability, H):
    grad_H = H.differentiate("altitude") * 1000
    cp = constants.cpv
    return -1 / (cp * rho * stability) * grad_H


def density_from_q(p, T, q):
    Rd = constants.dry_air_gas_constant
    Rv = constants.water_vapor_gas_constant
    return p / ((Rd + (Rv - Rd) * q) * T)


# %%
stability = (
    get_stability(
        lvl3.theta.interpolate_na("altitude", fill_value="extrapolate"),
        lvl3.ta.interpolate_na("altitude", fill_value="extrapolate"),
    )
    .coarsen(altitude=10)
    .mean()
    .rolling(altitude=10)
    .mean()
)
stability["altitude"] = stability["altitude"] + 5
stability.name = "stability"
# %%
rho = (
    density_from_q(
        lvl3.p.interpolate_na("altitude", fill_value="extrapolate"),
        lvl3.ta.interpolate_na("altitude", fill_value="extrapolate"),
        lvl3.q.interpolate_na("altitude", fill_value="extrapolate"),
    )
    .coarsen(altitude=10)
    .mean()
    .rolling(altitude=10)
    .mean()
)

rho["altitude"] = rho["altitude"] + 5
rho.name = "rho"
# %%#
# %%
H = (
    flux_data_arts2.cooling_rate.rolling(altitude=10).mean() * constants.cpv * rho
)  # J m-3 day-1
# %%
csc_stab = get_csc_stab(rho, stability, H)
csc_stab_of_mean = get_csc_stab(
    rho.mean("sonde_id"), stability.mean("sonde_id"), H.mean("sonde_id")
)

csc_cooling = get_csc_cooling(rho, stability, H)
csc_cooling_of_mean = get_csc_cooling(
    rho.mean("sonde_id"), stability.mean("sonde_id"), H.mean("sonde_id")
)

# %%
t = (
    lvl3.ta.interpolate_na("altitude", fill_value="extrapolate")
    .coarsen(altitude=10)
    .mean()
    .rolling(altitude=10)
    .mean()
)
t["altitude"] = t["altitude"] + 5


# %%
def get_hist_of_ta(da_t, da_var, bins_var, bins_ta=np.linspace(240, 305, 200)):
    bin_name = da_var.name + "_bin"
    hist = histogram(da_t, da_var, bins=[bins_ta, bins_var], dim=["altitude"]).compute()
    return (
        ((hist * hist[bin_name]).sum(dim=bin_name) / hist.sum(bin_name))
        .interpolate_na(dim="ta_bin")
        .rename(ta_bin="ta")
    )


histds = flux_data_arts2.sel(altitude=slice(0, 12000))
cr_bins = np.linspace(0, 5, 50)
cr_of_ta = get_hist_of_ta(histds.ta, histds.cooling_rate, cr_bins)
# %%
stab_bins = np.linspace(0, 8, 100)
stab_of_ta = get_hist_of_ta(t, stability, stab_bins)


# %%
csc_cooling.name = "csc_cooling"
csc_bins = np.linspace(-5, 5, 500)
csc_cooling_of_ta = get_hist_of_ta(t.isel(altitude=slice(0, -1)), csc_cooling, csc_bins)
csc_cooling_of_mean.name = "csc_cooling_of_mean"
csc_cooling_of_mean_of_ta = get_hist_of_ta(
    t.isel(altitude=slice(0, -1)), csc_cooling_of_mean, csc_bins
)
# %%
csc_stab.name = "csc_stability"
csc_bins = np.linspace(-5, 5, 500)
csc_stab_of_ta = get_hist_of_ta(t.isel(altitude=slice(0, -1)), csc_stab, csc_bins)
csc_stab_of_mean.name = "csc_stability_of_mean"
csc_stab_of_mean_of_ta = get_hist_of_ta(
    t.isel(altitude=slice(0, -1)), csc_stab_of_mean, csc_bins
)


# %%
def plot_all_east_west(da, ax, **kwargs):
    da.mean("sonde_id").plot(ax=ax, y="ta", label="all", **kwargs)
    da.where(lvl3.launch_lon < -40).mean("sonde_id").plot(
        ax=ax, y="ta", linestyle=":", label="West", **kwargs
    )
    da.where(lvl3.launch_lon > -40).mean("sonde_id").plot(
        ax=ax, y="ta", linestyle="--", label="East", **kwargs
    )
    return ax


# %%
sonde_subset = cs_sondes
title = "clear-sky sondes"
title2 = "non-mid-level clouds"

fig, allaxes = plt.subplots(
    ncols=3, nrows=2, figsize=(18, 12), sharey=True, sharex="col"
)
axes = allaxes[0, :]

plot_all_east_west(
    cr_of_ta.where(cr_of_ta.sonde_id.isin(sonde_subset)), ax=axes[0], color="k"
)
plot_all_east_west(
    stab_of_ta.where(stab_of_ta.sonde_id.isin(sonde_subset)), ax=axes[1], color="k"
)

plot_all_east_west(
    (csc_cooling_of_ta + csc_stab_of_ta).where(
        (csc_cooling_of_ta.sonde_id.isin(sonde_subset))
        & (csc_stab_of_ta.sonde_id.isin(sonde_subset))
    ),
    ax=axes[2],
    color="k",
)

axes = allaxes[1, :]
plot_all_east_west(
    cr_of_ta.where(~cr_of_ta.sonde_id.isin(ml_sondes)), ax=axes[0], color="k"
)
plot_all_east_west(
    stab_of_ta.where(~stab_of_ta.sonde_id.isin(ml_sondes)), ax=axes[1], color="k"
)
plot_all_east_west(
    (csc_cooling_of_ta + csc_stab_of_ta).where(
        ~(
            (csc_cooling_of_ta.sonde_id.isin(ml_sondes))
            & (csc_stab_of_ta.sonde_id.isin(ml_sondes))
        )
    ),
    ax=axes[2],
    color="k",
)
for ax in allaxes[:, 2]:
    ax.axvline(0, color="grey", alpha=0.5)
    ax.set_xlim(-0.5, 0.5)

for ax in allaxes.flatten():
    ax.invert_yaxis()
    ax.set_ylim(300, 240)
    ax.axhline(273.15, color="grey")
    ax.set_ylabel("")
    ax.axhline(267.8982991909455, color="grey", linestyle="--")
    ax.axhline(270.3643761588233, color="grey", linestyle=":")
allaxes[0, 0].legend()
allaxes[0, 1].set_title(title)
allaxes[1, 1].set_title(title2)
for ax in allaxes[:, 0]:
    ax.set_ylabel("Temperature / K")
allaxes[1, 0].set_xlabel("Cooling rate / K day$^{-1}$")
allaxes[1, 1].set_xlabel("Stability / K m$^{-1}$")
allaxes[1, 2].set_xlabel("Clear sky convergence / day$^{-1}$")
sns.despine(offset=10)
fig.savefig(
    "../plots/clear_sky_convergence_east_vs_west.pdf",
    dpi=300,
    bbox_inches="tight",
)

# %%
sonde_mask = ~(
    csc_cooling_of_ta.sonde_id.isin(ml_sondes)
)  # csc_cooling_of_ta.sonde_id.isin(cs_sondes) #

ds_cool = csc_cooling_of_ta.where(sonde_mask)
ds_stab = csc_stab_of_ta.where(sonde_mask)


fig, axes = plt.subplots(
    ncols=3,
    figsize=(12, 3),
    sharey=True,
)
ax = axes[0]
ds_cool.mean("sonde_id").plot(
    ax=ax, y="ta", color="C0", label=r"CSC$_\frac{\partial \mathcal{H}}{\partial z}$"
)
ds_stab.mean("sonde_id").rolling(ta=10).mean().plot(
    ax=ax,
    y="ta",
    color="C1",
    label=r"CSC$_\frac{\partial \mathcal{\sigma}}{\partial z}$",
)
(ds_cool + ds_stab).mean("sonde_id").rolling(ta=10).mean().plot(
    ax=ax, y="ta", color="k", label="Total"
)
ax.legend()
ax = axes[1]
ds_cool.where(lvl3.launch_lon < -40).mean("sonde_id").plot(ax=ax, y="ta", color="C0")
ds_stab.where(lvl3.launch_lon < -40).mean("sonde_id").rolling(ta=10).mean().plot(
    ax=ax, y="ta", color="C1"
)
(ds_cool + ds_stab).where(lvl3.launch_lon < -40).mean("sonde_id").rolling(
    ta=10
).mean().plot(
    ax=ax,
    y="ta",
    color="k",
)
ax.set_title("West")
ax.axhline(270.3643761588233, color="grey", alpha=0.5)
ax = axes[2]
ds_cool.where(lvl3.launch_lon > -40).mean("sonde_id").plot(ax=ax, y="ta", color="C0")
ds_stab.where(lvl3.launch_lon > -40).rolling(ta=10).mean().mean("sonde_id").plot(
    ax=ax, y="ta", color="C1"
)
(ds_cool + ds_stab).where(lvl3.launch_lon > -40).mean("sonde_id").rolling(
    ta=10
).mean().plot(
    ax=ax,
    y="ta",
    color="k",
)

ax.set_title("East")
ax.axhline(267.8982991909455, color="grey", alpha=0.5)


for ax in axes:
    ax.axvline(0, color="grey")
    ax.invert_yaxis()
    ax.set_ylim(300, 240)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("Clear sky convergence /  day$^{-1}$")
axes[0].set_ylabel("Temperature / K")

axes[1].set_title("West")
axes[1].axhline(270.3643761588233, color="grey", alpha=0.5)

sns.despine(offset=10)
fig.savefig(
    "../plots/clear_sky_convergence_no_ml_clouds.pdf",
    dpi=300,
    bbox_inches="tight",
)
# %%
# %% mean plots for cooling and stability
mask = lvl3.sonde_id.isin(ml_sondes)

csc_cooling_of_mean = get_csc_cooling(
    rho.where(mask).mean("sonde_id"),
    stability.where(mask).mean("sonde_id"),
    H.where(mask).mean("sonde_id"),
)
csc_stab_of_mean = get_csc_stab(
    rho.where(mask).mean("sonde_id"),
    stability.where(mask).mean("sonde_id"),
    H.where(mask).mean("sonde_id"),
)
fig, axes = plt.subplots(ncols=3, figsize=(18, 6), sharey=True)
mask = lvl3.sonde_id.isin(ml_sondes)
for ax, mask in zip(
    axes, [~mask, (mask & (lvl3.launch_lon < -40)), (mask & (lvl3.launch_lon > -40))]
):
    csc_cooling_of_mean = get_csc_cooling(
        rho.where(mask).mean("sonde_id"),
        stability.where(mask).mean("sonde_id"),
        H.where(mask).mean("sonde_id"),
    )
    csc_stab_of_mean = get_csc_stab(
        rho.where(mask).mean("sonde_id"),
        stability.where(mask).mean("sonde_id"),
        H.where(mask).mean("sonde_id"),
    )

    ax.plot(
        csc_cooling_of_mean,
        t.where(mask).mean("sonde_id").isel(altitude=slice(0, -1)),
        label="Cooling rate change",
    )
    ax.plot(
        csc_stab_of_mean.rolling(altitude=10).mean(),
        t.where(mask).mean("sonde_id").isel(altitude=slice(0, -1)),
        label="Cooling rate change",
    )
    ax.plot(
        (csc_stab_of_mean + csc_cooling_of_mean).rolling(altitude=10).mean(),
        t.where(mask).mean("sonde_id").isel(altitude=slice(0, -1)),
        label="Clear sky convergence",
        color="k",
    )
for ax in axes:
    ax.axvline(0, color="grey", alpha=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Clear sky convergence /  day$^{-1}$")
    ax.set_ylim(300, 240)
    ax.set_xlim(-0.5, 0.5)
axes[0].axhline(268.92767645, color="grey", alpha=0.5)
axes[2].axhline(267.8982991909455, color="grey", alpha=0.5)
axes[1].set_title("West")
axes[2].set_title("East")
axes[0].set_ylabel("Temperature / K")
axes[1].axhline(270.3643761588233, color="grey", alpha=0.5)
sns.despine(offset=10)
fig.savefig(
    "../plots/clear_sky_convergence_of_mean.pdf",
    dpi=300,
    bbox_inches="tight",
)
