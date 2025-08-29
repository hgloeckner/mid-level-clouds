# %%
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import cmocean as cmo
import seaborn as sns
import sys
import pydropsonde.helper.physics as pphysics

sys.path.append("../")
from myutils.constants_and_values import ml_sondes, cs_sondes
import myutils.open_datasets as open_datasets
from myutils.data_helper import get_hist_of_ta

# %%
east_color = "#c58040"
west_color = "#78a28e"

# %%
flux_data_arts2 = open_datasets.open_radiative_fluxes()

cid = open_datasets.get_cid()
lvl3 = open_datasets.open_dropsondes(
    f"{cid}/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr"
)
lvl4 = open_datasets.open_dropsondes(
    f"{cid}/HALO/dropsondes/Level_4/PERCUSION_Level_4.zarr"
)
# lvl3 = lvl3.where((lvl3.p_qc == 0) & (lvl3.ta_qc == 0) & (lvl3.rh_qc == 0), drop=True)

# %%
ds_cs = lvl3.sel(sonde_id=cs_sondes)

cmap = "BrBG"

lon_min, lon_max, lat_min, lat_max = -65, -15, 0, 23
fig, ax = plt.subplots(
    figsize=(12, 5.5), subplot_kw=dict(projection=ccrs.PlateCarree())
)
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    alpha=0.25,
    xlabel_style={"fontsize": 6},
    ylabel_style={"fontsize": 6},
)
gl.top_labels = False
gl.right_labels = False
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.add_feature(cf.LAND, zorder=0, edgecolor="black", facecolor="lightgrey")

ax.set_title("Surface wind direction")

p = ax.scatter(
    ds_cs.launch_lon.values,
    ds_cs.launch_lat.values,
    c=ds_cs.iwv.values,
    cmap=cmap,
)


# %%
def assign_circle_sonde_variable(dsl4):
    circle_idx = np.insert(dsl4.sondes_per_circle.cumsum(dim="circle_id").values, 0, 0)
    circle_ids = [
        np.repeat(
            dsl4.circle_id.isel(circle_id=i).values, (circle_idx[i + 1] - circle_idx[i])
        )
        for i, n in enumerate(circle_idx[:-1])
    ]
    return dsl4.assign_coords(
        {"circle_sondes": ("sonde_id", np.concatenate(circle_ids))}
    )


def find_nb_of_special_sondes_in_circle(dsl4, list_of_sondes):
    ml_circle_sondes = set(list_of_sondes) & set(dsl4.sonde_id.values)
    return (
        dsl4.sel(sonde_id=list(ml_circle_sondes))
        .swap_dims({"sonde_id": "sonde"})
        .sonde.groupby("circle_sondes")
        .count()
        .rename({"circle_sondes": "circle_id"})
    )


lev4 = assign_circle_sonde_variable(lvl4)
ml_lev4 = lev4.where(
    find_nb_of_special_sondes_in_circle(lev4, ml_sondes) > 4, drop=True
)
cs_lev4 = lev4.where(
    find_nb_of_special_sondes_in_circle(lev4, cs_sondes) > 6, drop=True
)


def integrate_in_p(da, p, alt_dim="altitude"):
    mask = (~np.isnan(da)) & (~np.isnan(p))
    zero_val = xr.DataArray(data=[0], dims=alt_dim, coords={alt_dim: [0]})
    pres_diff = xr.concat(
        [zero_val, p.where(mask, drop=True).sortby(alt_dim).diff(dim=alt_dim)],
        alt_dim,
    )
    return (-da.where(mask, drop=True).sortby(alt_dim) * pres_diff.T.values).cumsum(
        dim=alt_dim
    )


max_alt = 12000
lev4 = lev4.assign(
    iomega=lev4.omega.sel(altitude=slice(None, max_alt))
    .interpolate_na(dim="altitude")
    .fillna(0)
    .integrate(coord="altitude"),
    iomega_low=lev4.omega.sel(altitude=slice(None, 6000))
    .interpolate_na(dim="altitude")
    .fillna(0)
    .integrate(coord="altitude"),
    iomega_high=lev4.omega.sel(altitude=slice(6000, max_alt))
    .interpolate_na(dim="altitude")
    .fillna(0)
    .integrate(coord="altitude"),
    rho=pphysics.density_from_q(
        lev4.p_mean,
        lev4.ta_mean,
        lev4.q_mean,
    ),
    omega_x=integrate_in_p(
        lev4.u_dudx,
        lev4.p_mean,
        alt_dim="altitude",
    )
    * 0.01
    * 60**2,
    omega_y=integrate_in_p(
        lev4.v_dvdy,
        lev4.p_mean,
        alt_dim="altitude",
    )
    * 0.01
    * 60**2,
)
# %% omega directional

fig, axes = plt.subplots(figsize=(18, 5.5), ncols=3, sharey=True, sharex=True)
for var, style in [("omega_x", ":"), ("omega_y", "--"), ("omega", "-")]:
    lev4[var].mean(dim="circle_id").plot(
        y="altitude",
        ax=axes[0],
        color="k",
        linestyle=style,
        label=var,
    )
    lev4[var].where(lev4.circle_lon > -40).mean(dim="circle_id").plot(
        y="altitude",
        ax=axes[1],
        color=east_color,
        linestyle=style,
    )
    lev4[var].where(lev4.circle_lon < -40).mean(dim="circle_id").plot(
        y="altitude",
        ax=axes[2],
        color=west_color,
        linestyle=style,
    )
axes[0].legend()
for ax in axes:
    ax.axvline(0, color="k", alpha=0.5)
sns.despine(offset={"left": 10})
# %%
# %%


# %%# %%
ds = lev4
omega_bins = np.linspace(-80, 21, 100)
ta_bins = np.linspace(220, 302, 200)
omega_of_ta = get_hist_of_ta(ds.ta_mean, ds.omega, omega_bins, bins_ta=ta_bins)
div_bins = np.linspace(-10e-5, 21e-5, 100)
div_of_ta = get_hist_of_ta(ds.ta_mean, ds.div, div_bins, bins_ta=ta_bins)
div_of_ta.name = "div"
omega_of_ta.name = "omega"
rh_bins = np.linspace(0, 1.1, 100)
rh_of_ta = get_hist_of_ta(ds.ta_mean, ds.rh_mean, rh_bins, bins_ta=ta_bins)
rh_of_ta.name = "rh"
# %%
ds = xr.merge([omega_of_ta, div_of_ta, rh_of_ta]).assign(
    iomega=lev4.iomega,
    iomega_low=lev4.iomega_low,
    iomega_high=lev4.iomega_high,
    iwv_mean=lev4.iwv_mean,
    sfc_wspd_mean=lev4.wspd_mean.bfill("altitude").sel(altitude=0),
)
# %%# %%
var = "omega_x"  # "omega"
sort_var = "sfc_wspd_mean"
y_var = "altitude"  # "ta_mean"
plot_ds = lev4.assign(
    sfc_wspd_mean=lev4.wspd_mean.bfill("altitude").sel(altitude=0),
)
kwargs = dict(
    vmin=-10,
    vmax=10,
    cmap=cmo.cm.balance,
    # cmap='Blues',
)

fig, ax = plt.subplots(figsize=(24, 5.5))

plot_ds.sortby(sort_var)[var].plot(ax=ax, y=y_var, **kwargs)

east_mask = ds.sortby(sort_var).circle_lon > -40
colors = np.where(east_mask.values, east_color, west_color)
# ax.invert_yaxis()
# ax.axhline(267.5, color="k")
ax.tick_params(axis="x", labelrotation=90)
for xtick, color in zip(ax.get_xticklabels(), colors):
    xtick.set_color(color)
# %%

# %%
yvar = "iomega_low"
cvar = "iwv_mean"
kwargs = dict(
    vmin=36,
    vmax=60,
    cmap="RdYlBu",
)
factor = 6000
mask = ds.circle_lon > -40
fig, ax = plt.subplots(figsize=(5.5, 5.5))

p = ax.scatter(
    ds.iomega_high.where(mask) / factor,
    ds[yvar].where(mask) / factor,
    c=ds[cvar].where(mask),
    label="east",
    **kwargs,
)
ax.scatter(
    ds.iomega_high.where(~mask) / factor,
    ds[yvar].where(~mask) / factor,
    c=ds[cvar].where(~mask),
    marker="X",
    label="west",
    **kwargs,
)
cax = fig.add_axes((0.83, 0.2, 0.01, 0.25))
cb = fig.colorbar(p, cax=cax, ticks=[40, 48, 55, 60], extend="max")
ax.axvline(0, color="k", linestyle="-", alpha=0.5)
ax.axhline(0, color="k", linestyle="-", alpha=0.5)
ax.set_xlim(-20, 15)
ax.set_ylim(-20, 15)
ax.set_xlabel("omega above 6000m mean")
ax.set_ylabel("omega below 6000m mean")
# %%
downkwargs = dict(
    linestyle=":",
)
upkwargs = dict(
    linestyle="-",
)
var = "div"
split_var = "iomega"

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ds.where(ds[split_var] > 0)[var].mean("circle_id").rolling(ta_mean=10).mean().plot(
    y="ta_mean", label="down", c="k", **downkwargs
)
ds.where(ds[split_var] < 0)[var].mean("circle_id").rolling(ta_mean=10).mean().plot(
    y="ta_mean", label="up", c="k", **upkwargs
)


ds.where((ds[split_var] > 0) & (ds.circle_lon < -40))[var].mean("circle_id").rolling(
    ta_mean=10
).mean().plot(y="ta_mean", label="", c=west_color, **downkwargs)
ds.where((ds[split_var] < 0) & (ds.circle_lon < -40))[var].mean("circle_id").rolling(
    ta_mean=10
).mean().plot(y="ta_mean", label="West", c=west_color, **upkwargs)

ds.where((ds[split_var] > 0) & (ds.circle_lon > -40))[var].mean("circle_id").rolling(
    ta_mean=10
).mean().plot(y="ta_mean", label="", c=east_color, **downkwargs)
ds.where((ds[split_var] < 0) & (ds.circle_lon > -40))[var].mean("circle_id").rolling(
    ta_mean=10
).mean().plot(y="ta_mean", label="East", c=east_color, **upkwargs)


ax.legend()
ax.axvline(0, color="k", alpha=0.5)
ax.axhline(270, color="k", alpha=0.5)
ax.axhline(267.5, color="k", alpha=0.5)
ax.invert_yaxis()


# %%

fig, axes = plt.subplots(
    figsize=(12, 5),
    nrows=1,
    ncols=3,
    sharey=True,
    sharex=True,
)
var = "div"
split_var = "iomega"
for ax, split_var in zip(axes, ["iomega", "iomega_low", "iomega_high"]):
    (
        -ds.where((ds[split_var]) > 0)[var].mean("circle_id")
        + ds.where(ds[split_var] < 0)[var].mean("circle_id")
    ).rolling(ta_mean=10).mean().plot(ax=ax, y="ta_mean", label="all", c="k")

    (
        -ds.where(ds[split_var] > 0)[var].where(ds.circle_lon < -40).mean("circle_id")
        + ds.where(ds[split_var] < 0)[var].where(ds.circle_lon < -40).mean("circle_id")
    ).rolling(ta_mean=10).mean().plot(ax=ax, y="ta_mean", label="West", c=west_color)

    (
        -ds.where(ds[split_var] > 0)[var].where(ds.circle_lon > -40).mean("circle_id")
        + ds.where(ds[split_var] < 0)[var].where(ds.circle_lon > -40).mean("circle_id")
    ).rolling(ta_mean=10).mean().plot(ax=ax, y="ta_mean", label="East", c=east_color)
    ax.axvline(0, color="k", alpha=0.5)
    ax.axhline(270, color=west_color, alpha=0.5)
    ax.axhline(267.5, color=east_color, alpha=0.5)
    ax.invert_yaxis()
    ax.set_ylabel("")
    ax.set_xlabel("div")

axes[0].set_ylabel("circle mean temperature / K")
axes[0].set_title("Net divergence \nfrom updraft regions ")
axes[1].set_title("Net divergence \nfrom updraft below 6000m regions ")
axes[2].set_title("Net divergence \nfrom updraft above 6000m regions ")


axes[0].legend()
sns.despine(offset=10)
fig.savefig(
    "../plots/net_divergence_from_updrafts.pdf",
    bbox_inches="tight",
)


# %%
def plot_different_omega_regimes(ds, ax, mask=None, **kwargs):
    if mask is not None:
        ds = ds.where(mask)

    ds.where((ds.iomega_low > 0) & (ds.iomega_high < 0)).div.mean("circle_id").rolling(
        ta_mean=10
    ).mean().plot(
        ax=ax, y="ta_mean", linestyle="--", label="low down, high up", **kwargs
    )
    ds.where((ds.iomega_low > 0) & (ds.iomega_high > 0)).div.mean("circle_id").rolling(
        ta_mean=10
    ).mean().plot(ax=ax, y="ta_mean", linestyle=":", label="both down", **kwargs)
    ds.where((ds.iomega_low < 0) & (ds.iomega_high < 0)).div.mean("circle_id").rolling(
        ta_mean=10
    ).mean().plot(ax=ax, y="ta_mean", linestyle="-", label="both up", **kwargs)
    ds.where((ds.iomega_low < 0) & (ds.iomega_high > 0)).div.mean("circle_id").rolling(
        ta_mean=10
    ).mean().plot(
        ax=ax, y="ta_mean", linestyle="-.", label="low up, high down", **kwargs
    )


fig, axes = plt.subplots(figsize=(12, 5.5), ncols=3, sharex=True, sharey=True)

plot_different_omega_regimes(ds, axes[0], c="k")
mask = ds.circle_lon > -40
title = "East"

plot_different_omega_regimes(ds, axes[1], mask=mask, c=east_color)
axes[1].set_title(title)
mask = ds.circle_lon < -40
title = "West"

plot_different_omega_regimes(ds, axes[2], mask=mask, c=west_color)
axes[2].set_title(title)


axes[0].invert_yaxis()
axes[0].legend(loc=4)
for ax in axes:
    ax.set_ylabel("")
    ax.axvline(0, color="k", alpha=0.5)
    ax.axhline(270, color=west_color, alpha=0.5)
    ax.axhline(267.5, color=east_color, alpha=0.5)
axes[0].set_ylabel("circle mean temperature / K")
sns.despine(offset=10)
fig.savefig(
    "../plots/omega_regimes.pdf",
    bbox_inches="tight",
)
# %%
for mask, label in [
    (True, "all"),
    (ds.circle_lon > -40, "East"),
    (ds.circle_lon < -40, "West"),
]:
    print(
        label + " low down, high up",
        ds.where((ds.iomega_low > 0) & (ds.iomega_high < 0) & mask)
        # .iomega.count()
        .sfc_wspd_mean.mean("circle_id")
        .values,
    )
    print(
        label + " both down",
        ds.where((ds.iomega_low > 0) & (ds.iomega_high > 0) & mask)
        # .iomega.count()
        .sfc_wspd_mean.mean("circle_id")
        .values,
    )
    print(
        label + " both up",
        ds.where((ds.iomega_low < 0) & (ds.iomega_high < 0) & mask)
        # .iomega.count()
        .sfc_wspd_mean.mean("circle_id")
        .values,
    )
    print(
        label + " low up, high down",
        ds.where((ds.iomega_low < 0) & (ds.iomega_high > 0) & mask)
        # .iomega.count()
        .sfc_wspd_mean.mean("circle_id")
        .values,
    )
# %%
ds.where(
    (ds.iomega_low < 0) & (ds.iomega_high > 0) & mask, drop=True
).sfc_wspd_mean.plot(marker="o")


# %%

# %%
mask_both_down = (ds.iomega_low > 0) & (ds.iomega_high > 0)
mask_both_up = (ds.iomega_low < 0) & (ds.iomega_high < 0)
mask_mid_conv = (ds.iomega_low > 0) & (ds.iomega_high < 0)
mask_mid_div = (ds.iomega_low < 0) & (ds.iomega_high > 0)
var = "div"
fig, axes = plt.subplots(figsize=(11, 11), nrows=2, ncols=2, sharex=True, sharey=True)

for idx, (mask, label, linestyle) in enumerate(
    [
        (mask_both_down, "both down", ":"),
        (mask_both_up, "both up", "-"),
        (mask_mid_conv, "low down, high up", "--"),
        (mask_mid_div, "low up, high down", "-."),
    ]
):
    ax = axes.flatten()[idx]
    east_ds = ds.where(mask & (ds.circle_lon > -40), drop=True)
    west_ds = ds.where(mask & (ds.circle_lon < -40), drop=True)

    for circle in east_ds.circle_id.values:
        east_ds.sel(circle_id=circle)[var].rolling(ta_mean=10).mean().plot(
            ax=ax,
            y="ta_mean",
            c=east_color,
            alpha=0.2,
        )
    east_ds.mean("circle_id")[var].rolling(ta_mean=10).mean().plot(
        ax=ax,
        y="ta_mean",
        c=east_color,
        linewidth=2,
    )
    for circle in west_ds.circle_id.values:
        west_ds.sel(circle_id=circle)[var].rolling(ta_mean=10).mean().plot(
            ax=ax,
            y="ta_mean",
            c=west_color,
            alpha=0.2,
        )
    west_ds.mean("circle_id")[var].rolling(ta_mean=10).mean().plot(
        ax=ax,
        y="ta_mean",
        c=west_color,
        linewidth=2,
    )
    ax.set_title(label)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.axvline(0, color="k", alpha=0.5)
axes[0, 0].invert_yaxis()
axes[1, 0].set_ylabel("circle mean temperature / K")
axes[1, 0].set_xlabel("divergence / s$^{-1}$")

axes[0, 0].set_ylabel("circle mean temperature / K")
axes[1, 1].set_xlabel("divergence / s$^{-1}$")
axes[0, 0].set_xlim(-5e-5, 5e-5)
sns.despine(offset=10)
# %%

basile_omega = xr.open_dataset(
    "/work/bb1153/b381959/ORCESTRA/omega_ORCESTRA_new.zarr",
    engine="zarr",
    chunks={},
)

# %%
ds = basile_omega.assign_coords(
    lat=(
        "lat",
        xr.where(basile_omega.lats < -100, np.nan, basile_omega.lats)
        .mean(dim="lon")
        .values,
    ),
    lon=(
        "lon",
        xr.where(basile_omega.lons < -100, np.nan, basile_omega.lons)
        .mean(dim="lat")
        .values,
    ),
)

# %%

stacked = (
    ds.isel(lon=slice(200, 1300), lat=slice(130, 950))
    .stack(point=("lat", "lon"))
    .set_coords(["lats", "lons"])
    .sel(channel=0, time="2024-09-14")
)
# %%
plt.scatter(
    stacked.T_em,
    stacked.omega,
    s=2,
)
# %%

wv, no_wv = open_datasets.open_wales(masked=True)
# %%
ml_min = 4000
ml_max = 8000


def get_ml_cloud(wales):
    mid_level = wales.sel(altitude=slice(ml_min, ml_max))

    return (
        wales.where((mid_level.bsrgl.max(dim="altitude") >= 20))
        .compute()
        .dropna("time", how="all")
    )


def find_Tem(
    ds,
    new_var_name="alt_em",
    variable_name="tau2gl",
    threshold_max=1.1,
    threshold_min=0.9,
):
    ds = ds.sortby("altitude").chunk({"altitude": -1, "time": 1000})
    mask = (ds[variable_name] > threshold_min) & (ds[variable_name] < threshold_max)
    mask_inv = mask.isel(altitude=slice(None, None, -1))
    highest_altitude = ds.altitude.values[
        mask.sizes["altitude"] - 1 - mask_inv.argmax(dim="altitude")
    ]
    return xr.Dataset(
        data_vars={
            new_var_name: (
                ("time",),
                np.where(mask.any(dim="altitude"), highest_altitude, np.nan),
            ),
            f"{new_var_name}_ta": (
                ("time",),
                ds.ta.interp(altitude=highest_altitude).values[0, :],
            ),
        },
        coords={"time": ds.time},
    )


ml_clouds = get_ml_cloud(no_wv)
# %%
