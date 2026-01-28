# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean.cm as cmo
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import ListedColormap, Normalize
import sys

sys.path.append("../")
from myutils import open_datasets as od
import myutils.data_helper as dh

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib as mpl

regions = {
    "north": dh.north,
    "east": dh.east,
    "west": dh.west,
}
# %%

wv, no_wv = od.open_wales(masked=False)

cid = "ipns://latest.orcestra-campaign.org"  # open_datasets.get_cid()
dropsondes = xr.open_dataset(
    f"{cid}/products/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr",
    engine="zarr",
)
rdata = xr.open_dataset(
    "ipfs://bafybeifxtmq5mpn7vwiiwl4vlpoil7rgm2tnhmkeyqsyudleqegxzvwl3a", engine="zarr"
)

rdata = (
    rdata.assign(reflectivity=(10 * np.log(rdata.radar_reflectivity)))
    .sel(time=slice(np.datetime64("2024-08-10"), np.datetime64("2024-09-29")))
    .sortby("altitude")
)
no_wv = no_wv.sel(time=slice(np.datetime64("2024-08-10"), np.datetime64("2024-09-29")))
no_wv = xr.where(no_wv.bsrgl_flags == 8, 100, no_wv)
no_wv = no_wv.where((no_wv.bsrgl_flags == 0) | (no_wv.bsrgl_flags == 8)).sortby(
    "altitude"
)

# %%
wales = {}
radar = {}
wales["full"] = xr.DataArray(
    data=dh.find_highest_cloud_altitude(no_wv.sel(altitude=slice(200, None))),
    dims="time",
    name="cloud-top",
)
radar["full"] = xr.DataArray(
    data=dh.find_highest_cloud_altitude(
        rdata.sel(altitude=slice(200, None)),
        variable_name="reflectivity",
        threshold=-1e3,
    ),
    dims="time",
    name="cloud-top",
)
# %%
radar["full"] = radar["full"].assign_coords(
    time=rdata.time, longitude=rdata.longitude, latitude=rdata.latitude
)
wales["full"] = wales["full"].assign_coords(
    time=no_wv.time, longitude=no_wv.longitude, latitude=no_wv.latitude
)
# %%
sns.set_context("paper", font_scale=1.3)
fig, ax = plt.subplots(figsize=(8, 5))
histkwargs = {
    "stat": "density",
    "bins": 30,
    "binrange": (0, 14000),
    "element": "step",
    "fill": False,
    "linewidth": 1,
    "kde": False,
    "alpha": 0.5,
    "ax": ax,
    "y": "cloud-top",
}
sns.histplot(wales["full"].to_dataframe().dropna(), color="k", **histkwargs)
sns.histplot(
    radar["full"].to_dataframe().dropna(), color="k", linestyle="--", **histkwargs
)

kdekwargs = {
    "fill": False,
    "linewidth": 2,
    "ax": ax,
    "y": "cloud-top",
}
pt = sns.kdeplot(
    wales["full"].to_dataframe().dropna(),
    color="k",
    label="Lidar Cloud-Top",
    **kdekwargs,
)
pt = sns.kdeplot(
    radar["full"].to_dataframe().dropna(),
    color="k",
    linestyle="--",
    label="Radar Cloud-Top",
    **kdekwargs,
)
"""
for c, name in zip(["#FF7982", "#B6001E", "#00b4d8"], ["north", "east", "west"]):
    sns.kdeplot(
        dh.sel_sub_domain(
            wales["full"],
            regions[name],
            item_var="time",
            lon_var="longitude",
            lat_var="latitude",
        ).to_dataframe().dropna(),
        color=c,
        label=f"Wales {name.capitalize()}",
        **kdekwargs,
    )
    sns.kdeplot(
        dh.sel_sub_domain(
            radar["full"],
            regions[name],
            item_var="time",
            lon_var="longitude",
            lat_var="latitude",
        ).to_dataframe().dropna(),
        color=c,
        linestyle="--",
        label=f"Radar {name.capitalize()}",
        **kdekwargs,
    )
"""

ax.axhline(5799.8, xmax=0.5, color="k", alpha=0.5)
ax.axhline(5859.83, xmax=0.5, color="k", linestyle="--", alpha=0.5)
ax.axhline(6584.25, xmax=0.5, color="#FF7982", alpha=0.5, label="North", linewidth=2)
ax.axhline(6363.27, xmax=0.5, linestyle="--", color="#FF7982", alpha=0.5, linewidth=2)
ax.axhline(5817.79, xmax=0.5, color="#B6001E", alpha=0.5, label="East", linewidth=2)
ax.axhline(6005.63, xmax=0.5, linestyle="--", color="#B6001E", alpha=0.5, linewidth=2)
ax.axhline(5389.24, xmax=0.5, color="#00b4d8", alpha=0.5, label="West", linewidth=2)
ax.axhline(5320.47, xmax=0.5, linestyle="--", color="#00b4d8", alpha=0.5, linewidth=2)

ax.set_yticks(
    [
        0,
        2000,
        4000,
        5460,
        5819,
        6560,
        8000,
        10000,
        12000,
        14000,
    ]
)
sns.despine()
ax.legend()
ax.set_ylim(0, 14000)
ax.set_ylabel("Cloud Top Altitude / m")
fig.tight_layout()
fig.savefig("plots/cloud_top_altitude_distribution.pdf")
fig.savefig(
    "/scratch/m/m301046/cloud_top_altitude_distribution_mean.pdf", transparent=True
)
# %%
"""
region = [
    "full",
    "full",
    "north",
    "north",
    "east",
    "east",
    "west",
    "west",
]
instrument = ["wales", "radar"] * 4
for idx, i in enumerate(np.arange(start=2, stop=10, step=1)):
    x, y = pt.lines[i].get_data()
    idx4000 = np.argmin(np.abs(y - 4000))
    idx8000 = np.argmin(np.abs(y - 9000))
    midmax = np.argmax(x[idx4000:idx8000]) + idx4000
    print(region[idx], instrument[idx])
    print(f"Max mid level cloud top altitude for region {i}: {y[midmax]:.2f} m")

for region, height in zip(
    ["north", "east", "west"],
    [6559.80, 5819.43, 5378.91],
):
    ds = dh.sel_sub_domain(
        dropsondes,
        regions[region],
        item_var="sonde",
        lon_var="launch_lon",
        lat_var="launch_lat",
    )
    print(
        "ct temperature for",
        region,
        ": ",
        ds.ta.sel(altitude=height, method="nearest").mean("sonde").values,
        "K",
    )
"""
# %% cloud top map plot


norm = mpl.colors.BoundaryNorm(
    boundaries=(0, 4000, 6000, 8000, 15000),
    ncolors=4,
)
cmap = mpl.colors.ListedColormap(
    ["#E2E1E2", "#FFC929", "#FF7E15", "#383637"]
)  # "twilight"#
cm = 1 / 2.54  # centimeters to inches
cw = 20 * cm
fig, axes = plt.subplots(
    ncols=2, figsize=(cw, cw * 0.5), subplot_kw={"projection": ccrs.Robinson()}
)

rcoars = radar["full"].reset_coords().resample(time="2S").nearest(tolerance="1S")
wcoars = (
    wales["full"]
    .reset_coords()
    .resample(time="2S")
    .nearest(tolerance="1S")
    .interp(time=rcoars.time)
)
coars = xr.DataArray(
    data=np.nanmax(
        np.stack([rcoars["cloud-top"].values, wcoars["cloud-top"].values]), axis=0
    ),
    dims="time",
    name="cloud-top",
).assign_coords(
    time=rcoars.time,
    longitude=rcoars.longitude,
    latitude=rcoars.latitude,
)

regions = {
    "east": dh.east,
    "west": dh.west,
}

for idx, (name, region) in enumerate(regions.items()):
    plt_ds = dh.sel_sub_domain(
        coars, region, lat_var="latitude", lon_var="longitude", item_var="time"
    )
    p = axes[idx].scatter(
        plt_ds.longitude,
        plt_ds.latitude,
        c=plt_ds,
        transform=ccrs.PlateCarree(),
        alpha=0.5,
        s=0.5,
        cmap=cmap,
        norm=norm,
    )
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.95, 0.3, 0.02, 0.4])
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)

for ax in axes:
    ax.add_feature(cf.COASTLINE, linewidth=0.5)
fig.savefig("/scratch/m/m301046/cloud_top_map_max.pdf")
# %%
coars.to_zarr(
    "/scratch/m/m301046/wales_radar_cloud_top_max.zarr",
    mode="w",
    zarr_format=2,
)
# %%

date = "2024-08-21"
dw = no_wv.sortby("altitude").sel(time=date).bsrgl.load()
dr = (
    rdata.sortby("altitude")
    .sel(time=date, altitude=slice(200, None))
    .reflectivity.load()
)
dd = rdata.sortby("altitude").sel(time=date, altitude=slice(200, None))

dsdate = dropsondes.swap_dims({"sonde": "launch_time"}).sel(launch_time=date)


# %% one day radar and wales

fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(12, 8))

cmap = cmo.delta_r
norm = TwoSlopeNorm(vcenter=20, vmax=80)

rr = dr.plot(
    y="altitude",
    ax=axes[0],
    norm=norm,
    cmap=cmap,
    add_colorbar=False,
    rasterized=True,
)


def two_part_cmap(cmap_lo, cmap_hi, vmin, split, vmax, N=256):
    f = (split - vmin) / (vmax - vmin)
    n_lo = max(1, int(round(N * f)))
    n_hi = max(1, N - n_lo)
    c_lo = cmap_lo(np.linspace(0, 1, n_lo))
    c_hi = cmap_hi(np.linspace(0, 1, n_hi))
    return ListedColormap(np.vstack([c_lo, c_hi]))


def cmap_segment(cmap, a=0.0, b=1.0, N=256):
    # a,b in [0,1]: 0–0.5 = lower half, 0.5–1 = upper half
    cmap = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    return ListedColormap(cmap(np.linspace(a, b, N)))


vmin, vmax, split = 0, 100, 20
cmap = two_part_cmap(cmo.turbid, cmo.deep, vmin, split, vmax)
norm = Normalize(vmin=vmin, vmax=vmax)

bsw = dw.plot(
    y="altitude",
    ax=axes[1],
    norm=norm,
    cmap=cmap,
    add_colorbar=False,
    rasterized=True,
)

cmap = cmo.delta
norm = TwoSlopeNorm(vcenter=-1, vmax=5, vmin=-6)
vel = dd.radar_doppler_velocity_corrected_no_wind.plot(
    y="altitude",
    ax=axes[2],
    cmap=cmap,
    norm=norm,
    add_colorbar=False,
    rasterized=True,
)

for lt in dsdate.launch_time.values:
    ds = dsdate.sel(launch_time=lt).sel(altitude=slice(4000, 7000))
    ds = (
        ds.interpolate_na(dim="altitude", fill_value="extrapolate")
        .swap_dims({"altitude": "ta"})
        .sortby("ta")
    )
    ta, idx = np.unique(ds.ta, return_index=True)
    ds = ds.isel(ta=idx).sortby("ta")
    try:
        ds = ds.interp(ta=273.15)
    except ValueError:
        continue
    else:
        for ax in axes:
            ax.plot(
                ds.launch_time,
                ds.altitude,
                marker="o",
                markersize=2,
                color="k",
            )

for ax in axes:
    ax.set_ylabel("Altitude / m")
    ax.set_xlabel("")
axes[2].set_xlabel("Time / UTC")

sns.despine(offset={"left": 10})
cvel = fig.colorbar(vel, ax=axes[2], extend="both", ticks=[-5, -1, 1, 5], fraction=0.08)
cvel.ax.set_ylabel("Doppler Velocity / m s$^{-1}$", fontsize=11)
cbs = fig.colorbar(bsw, ax=axes[1], extend="both", fraction=0.08)
cbs.ax.set_ylabel("Backscatter Ratio / 1", fontsize=11)

crr = fig.colorbar(
    rr, ax=axes[0], extend="max", ticks=[-150, -50, 0, 20, 50], fraction=0.08
)
crr.ax.set_ylabel("Reflectivity / dBZ", fontsize=11)
fig.savefig(f"plots/clouds_{date}.pdf", dpi=300)
