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
from myutils import open_datasets
from myutils import physics_helper as physics
from myutils.data_helper import sel_sub_domain
import myutils.data_helper as dh
# %%

wv, no_wv = open_datasets.open_wales(masked=False)
wv = wv.assign(q=physics.wv2q(wv))

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

# %%
wales = {}
radar = {}
wales["full"] = xr.DataArray(
    data=dh.find_highest_cloud_altitude(no_wv.sel(altitude=slice(200, None))),
    dims="time",
    name="cloud-top",
).to_dataframe()
radar["full"] = xr.DataArray(
    data=dh.find_highest_cloud_altitude(
        rdata.sel(altitude=slice(200, None)),
        variable_name="reflectivity",
        threshold=-1e3,
    ),
    dims="time",
    name="cloud-top",
).to_dataframe()

# %%
for region, name in zip(
    [dh.east, dh.west, dh.north],
    ["east", "west", "north"],
):
    wales[name] = xr.DataArray(
        data=dh.find_highest_cloud_altitude(
            sel_sub_domain(
                no_wv, region, item_var="time", lon_var="longitude", lat_var="latitude"
            )
        ),
        dims="time",
        name="cloud-top",
    ).to_dataframe()
    radar[name] = xr.DataArray(
        data=dh.find_highest_cloud_altitude(
            sel_sub_domain(
                rdata.sel(altitude=slice(200, None)),
                region,
                item_var="time",
                lon_var="longitude",
                lat_var="latitude",
            ),
            variable_name="reflectivity",
            threshold=-1e3,
        ),
        dims="time",
        name="cloud-top",
    ).to_dataframe()

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
sns.histplot(wales["full"].dropna(), color="k", **histkwargs)
sns.histplot(radar["full"].dropna(), color="k", linestyle="--", **histkwargs)

kdekwargs = {
    "fill": False,
    "linewidth": 2,
    "ax": ax,
    "y": "cloud-top",
}
pt = sns.kdeplot(
    wales["full"].dropna(), color="k", label="Lidar Cloud-Top", **kdekwargs
)
pt = sns.kdeplot(
    radar["full"].dropna(),
    color="k",
    linestyle="--",
    label="Radar Cloud-Top",
    **kdekwargs,
)
"""
for c, name in zip(["#FF7982", "#B6001E", "#00b4d8"], ["north", "east", "west"]):
    sns.kdeplot(
        wales[name].dropna(),
        color=c,
        label=f"Wales {name.capitalize()}",
        **kdekwargs,
    )
    sns.kdeplot(
        radar[name].dropna(),
        color=c,
        linestyle="--",
        label=f"Radar {name.capitalize()}",
        **kdekwargs,
    )
"""
ax.axhline(5799.8, xmax=0.5, color="k", alpha=0.5)
ax.axhline(5859.83, xmax=0.5, color="k", linestyle="--", alpha=0.5)
ax.axhline(6559.85, xmax=0.5, color="#FF7982", alpha=0.5, label="North", linewidth=2)
ax.axhline(6363.27, xmax=0.5, linestyle="--", color="#FF7982", alpha=0.5, linewidth=2)
ax.axhline(5819.00, xmax=0.5, color="#B6001E", alpha=0.5, label="East", linewidth=2)
ax.axhline(6005.63, xmax=0.5, linestyle="--", color="#B6001E", alpha=0.5, linewidth=2)
ax.axhline(5460.19, xmax=0.5, color="#00b4d8", alpha=0.5, label="West", linewidth=2)
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
    ds = sel_sub_domain(
        dropsondes,
        eval(region),
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
# %%

# %%
fig, ax = plt.subplots(figsize=(8, 5))


(no_wv.bsrgl.where(no_wv.bsrgl > 20).count(dim="time") / no_wv.sizes["time"]).plot(
    y="altitude", label="lidar"
)
(
    rdata.reflectivity.where(rdata.reflectivity > -1e3).count(dim="time")
    / rdata.sizes["time"]
).sel(altitude=slice(200, None)).plot(y="altitude", label="radar")
ax.legend()

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

# %%

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

# %%


rd = rdata.where(rdata.reflectivity >= -1e3)
ld = no_wv.where(no_wv.bsrgl >= 20)
# %%

# %%
print("radar", rd.dropna(dim="time", how="all").sizes)
print("lidar", ld.dropna(dim="time", how="all").sizes)

# %%


# %%
wales = xr.where(no_wv.bsrgl >= 20, 1, np.nan)
# %%
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
sns.histplot(radar["full"].dropna(), **histkwargs)
sns.histplot(wales["full"].dropna(), **histkwargs)
sns.kdeplot(radar["full"].dropna(), **histkwargs)
sns.kdeplot(wales["full"].dropna(), **histkwargs)
