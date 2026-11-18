# %%
import pandas as pd

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage as ndimage
from xhistogram.xarray import histogram
import os, certifi

os.environ["SSL_CERT_FILE"] = certifi.where()  # set before opening the dataset

scratch_path = "/scratch/m/m301046/"
scratch_path = "/Users/helene/Documents/scratch/"


def get_labels_by_segment(rdata, max_gap, datavar="reflectivity"):
    labels = np.full((rdata.sizes["time"], rdata.sizes["altitude"]), np.nan)
    structure = ndimage.generate_binary_structure(2, 2)
    gaps = np.diff(rdata.time) > max_gap
    segments = np.concatenate(([0], np.cumsum(gaps)))
    offset = 0
    for id in np.unique(segments):
        seg_data = rdata.isel(time=np.where(segments == id)[0])
        lab, n = ndimage.label(
            seg_data[datavar].interpolate_na("altitude", max_gap=90).notnull(),
            structure=structure,
        )
        lab = np.where(lab > 0, lab + offset, np.nan)
        idxs = np.where(segments == id)[0]
        labels[idxs, :] = lab
        offset = offset + n
    return labels


def mask_small_objects(seg, min_size):
    sizes = np.bincount(np.nan_to_num(seg).astype(int).ravel())
    valid_id = np.flatnonzero(sizes > min_size)
    return np.where(np.isin(seg, valid_id), seg, np.nan)


# %%
rdata = xr.open_dataset(
    # "ipfs://bafybeifxtmq5mpn7vwiiwl4vlpoil7rgm2tnhmkeyqsyudleqegxzvwl3a",
    "ipfs://bafybeigmd3dovwm45ylfqxnn2jphsrdjl2jt3dfytv7grkyhleaq42jthe",
    engine="zarr",
).rename({"height": "altitude"})


# %%
reflectivity_var = "Zg"
# radar_reflectivity
rdata = (
    rdata.assign(
        reflectivity=(10 * np.log10(rdata[reflectivity_var])),
        # snr_db=10 * np.log10(rdata.radar_signal_to_noise),
    )
    .sel(time=slice(np.datetime64("2024-08-11"), np.datetime64("2024-09-29")))
    .sortby("altitude")
    .sel(altitude=slice(200, None))
    .chunk({"altitude": -1})
)
# rdata = rdata.where(rdata.snr_db > -17)
# %%

wales = (
    xr.open_dataset(
        "https://swift.dkrz.de/v1/dkrz_41caca03ec414c2f95f52b23b775134f/wales/wales_no_wv.zarr",
        chunks={},
    )
    .sel(time=slice(np.datetime64("2024-08-11"), np.datetime64("2024-09-29")))
    .sel(altitude=slice(200, None))
)

wales = wales.where(
    ((wales.bsrgl_flags == 0) & (wales.bsrgl >= 20)) | (wales.bsrgl_flags == 8)
)
# %%
walesseg = get_labels_by_segment(
    wales.chunk({"altitude": -1}), np.timedelta64(5, "m"), datavar="bsrgl"
)
walesbig = mask_small_objects(walesseg, 3)
# %%
radarseg = get_labels_by_segment(rdata, np.timedelta64(5, "m"))
radarbig = mask_small_objects(radarseg, 3)


# %%
def get_cloud_stats(seg, timeds):
    ti, ai = np.nonzero(~np.isnan(seg))
    df = pd.DataFrame(
        {
            "label": seg[ti, ai].astype(int),
            "ti": ti,
            "altitude": timeds.altitude.values[ai],
        }
    )

    stats = (
        df.groupby(["label", "ti"])["altitude"]
        .agg(cloud_base="min", cloud_top="max")
        .reset_index()
    )
    stats["step"] = stats["ti"] - stats.groupby("label")["ti"].transform("min")
    stats["time_idx"] = timeds["time"].values[stats["ti"]]

    ds = stats.set_index(["label", "step"])[
        ["cloud_top", "cloud_base", "time_idx"]
    ].to_xarray()
    ds = ds.assign(
        cloud_depth=ds.cloud_top - ds.cloud_base,
    )
    return stats, ds


radarstats, radarprops = get_cloud_stats(radarbig, rdata)
walesstats, walesprops = get_cloud_stats(walesbig, wales)
# %%
maxct = {}
for stats in [("radar", radarstats), ("wales", walesstats)]:
    maxct[stats[0]] = stats[1].groupby("time_idx")["cloud_top"].max().to_xarray()
# %%
# maxct["radar"].to_netcdf(scratch_path + "radar_max_cloud_top.nc")

# %%
altbins = np.linspace(0, 15000, 100)
hists = {}
for key in maxct.keys():
    hists[key] = histogram(
        maxct[key],
        dim=["time_idx"],
        bins=altbins,
    ).load()

# %%
specmacs = xr.open_dataset(
    "ipfs://bafybeieuwgibicmq3wkewlhdpvyqan32aiepktum27dsh4gx5kehxc2jsm",
    engine="zarr",
    chunks={},
).height_above_geoid.sel(time=slice("2024-08-11", "2024-09-29"))
hists["specmacs"] = histogram(
    specmacs,
    dim=["time"],
    bins=altbins,
    block_size=10,
).load()

# %%
hists["specmacs"] = (
    hists["specmacs"]
    .assign_coords(altitude=hists["specmacs"].height_above_geoid_bin / 1000)
    .swap_dims({"height_above_geoid_bin": "altitude"})
)
hists["wales"] = (
    hists["wales"]
    .assign_coords(altitude=hists["wales"].cloud_top_bin / 1000)
    .swap_dims({"cloud_top_bin": "altitude"})
)

# %%

border = 8

scale_factor = (
    hists["specmacs"].sel(altitude=slice(1, border)).sum()
    / hists["wales"].sel(altitude=slice(0, border)).sum()
).values
hists["mod_specmacs"] = xr.concat(
    [
        hists["specmacs"].sel(altitude=slice(0, border)).reset_coords(drop=True),
        hists["wales"].sel(altitude=slice(border, None)).reset_coords(drop=True)
        * scale_factor,
    ],
    dim="altitude",
)
# %%

l3ds = xr.open_dataset(
    "ipfs://bafybeiczbv7mycr2jois6t4dq3zwiltycomwo5xxvjqcjz2ot3newzar6q", engine="zarr"
)
l3ds = l3ds.swap_dims({"sonde": "launch_time"}).sel(
    launch_time=slice("2024-08-10", "2024-09-29")
)

# %%


def to_hourly(f):
    t = pd.date_range(
        pd.Timestamp(f.launch_time.min().values).ceil("h"),
        pd.Timestamp(f.launch_time.max().values),
        freq="1h",
    )
    return (
        f.reindex(launch_time=np.union1d(f.launch_time, t))
        .interpolate_na(dim="launch_time", method="pchip")
        .sel(launch_time=t)
        .assign_coords(flight_id=f.flight_id.values[0])
    )


tads = (
    l3ds.ta.interpolate_na(dim="altitude")
    .rolling(altitude=50, center=True)
    .mean()
    .assign_coords(flight_id=l3ds.flight_id)
    .groupby(l3ds.flight_id)
    .map(to_hourly)
)


# %%
ds = radarprops.assign(mean_time=radarprops.time_idx.groupby("label").mean("step"))
ds = ds.assign(
    mean_latitude=(
        "label",
        rdata.lat.sel(time=ds.mean_time, method="nearest").values,
    ),
    cloud_top_ta=(
        ("label", "step"),
        tads.reset_coords()
        .ta.sel(
            launch_time=ds.mean_time.broadcast_like(ds.time_idx),
            altitude=ds.cloud_top.fillna(0),
            method="nearest",
        )
        .values,
    ),
)


# %%
east = ds.where(ds.mean_time < np.datetime64("2024-09-05"), drop=True)
west = ds.where(ds.mean_time > np.datetime64("2024-09-07"), drop=True)

north = east.where(east.mean_latitude > 13.5, drop=True)
east = east.where(east.mean_latitude < 13.5, drop=True)


# %%
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib as mpl
import myutils.data_helper as dh

cmap = mpl.colors.ListedColormap(["#D3D3D3", "#D8782F", "#982B00", "#474747"])
bounds = [0, 4000, 6000, 8000, 15000]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

bins = np.linspace(200, 300, 50)  # np.linspace(0, 15000, 50)
ctvar = "cloud_top_ta"

sns.set_context("paper")
cw = 8.3
fig = plt.figure(figsize=(cw, cw), layout="constrained")
gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.4])

map_ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
map_ax.set_extent([-63, -15, 0, 22])
rect_ax = fig.add_subplot(gs[0, 1])

bottom = gs[1, :].subgridspec(1, 2)
ax_left = fig.add_subplot(bottom[0])
ax_right = fig.add_subplot(bottom[1])
"""
"""
im = map_ax.scatter(
    rdata.lon.sel(time=maxct["radar"].time_idx),
    rdata.lat.sel(time=maxct["radar"].time_idx),
    c=maxct["radar"],
    s=1e-3,
    cmap=cmap,
    norm=norm,
    rasterized=True,
    transform=ccrs.PlateCarree(),
)
cax = map_ax.inset_axes([0.88, 0.05, 0.02, 0.4])
cbar = fig.colorbar(im, cax=cax, orientation="vertical")
cbar.ax.set_yticks(bounds, labels=[int(bound / 1000) for bound in bounds], fontsize=10)
cbar.ax.set_ylabel("cloud top / km", fontsize=10, labelpad=1)
mapxticks = []
mapyticks = []
for region, label, xy in [
    (dh.east, "East", (-35.5, 12)),
    (dh.west, "West", (-45, 6.5)),
    (dh.north, "North", (-20, 17)),
]:
    map_ax.plot(region[:2, 0], region[:2, 1], color="k", lw=1, ls="solid")
    map_ax.plot(region[2:, 0], region[2:, 1], color="k", lw=1, ls="solid")
    map_ax.annotate(f"{label}", xy=xy, fontsize=8)
    mapxticks.append(region[:2, 0])
    mapyticks.append(region[1:-1, 1])
map_ax.set_xticks(np.concatenate(mapxticks))
map_ax.set_yticks(np.unique(np.concatenate(mapyticks)))
map_ax.set_xlabel("longitude / °E", fontsize=10)
map_ax.set_ylabel("latitude / °N", fontsize=10)
altbinskm = np.linspace(0, 15, 100)

cnts, values, bars = rect_ax.hist(
    maxct["radar"] / 1000,
    orientation="horizontal",
    bins=altbinskm,
    density=True,
)
for cnt, value, bar in zip(cnts, values, bars):
    bar.set_facecolor(cmap(norm(value * 1000), alpha=0.5))

for idx in range(len(bounds) - 1):
    for key, label, ls in [
        ("specmacs", "specMACS", "-"),
        ("wales", "Wales", ":"),
        ("mod_specmacs", "mod specMACS", "--"),
    ]:
        normed_hist = hists[key] / (hists[key].sum() * np.diff(altbinskm)[0])
        (
            normed_hist.sel(altitude=slice(bounds[idx] / 1000, bounds[idx + 1] / 1000))
        ).plot(
            ax=rect_ax,
            y="altitude",
            color=cmap(norm((bounds[idx] + bounds[idx + 1]) / 2)),
            lw=1,
            ls=ls,
            label=label,
        )
rect_ax.set_ylabel("cloud top / km", fontsize=10, labelpad=-2)
map_ax.coastlines()
rect_ax.set_xlim(0, 0.4)
rect_ax.set_xlabel("pdf")


handles = [
    mpl.lines.Line2D([], [], color=cmap(5), ls=":", lw=1, label="WALES"),
    mpl.lines.Line2D([], [], color=cmap(5), ls="-", lw=1, label="specMACS"),
    mpl.lines.Line2D([], [], color=cmap(5), ls="--", lw=1, label="specMACS (mod)"),
    mpl.patches.Patch(color=cmap(5), label="radar"),
]
rect_ax.legend(handles=handles, fontsize=7)


# lower plot
for pltds, label, color in [
    (north, "north", "#FF7982"),
    (west, "west", "#0085db"),
    (east, "east", "#960018"),
]:
    top = histogram(
        pltds[ctvar],
        dim=["label", "step"],
        bins=bins,
        block_size=1,
    )
    lowtop = histogram(
        pltds.where(pltds.cloud_base < 1500)[ctvar],
        dim=["label", "step"],
        bins=bins,
        block_size=1,
    )
    ax_left.stairs(
        (top / top.sum(f"{ctvar}_bin")),
        edges=bins,
        orientation="horizontal",
        color=color,
        label=label,
    )
    ax_right.stairs(
        (lowtop / lowtop.sum(f"{ctvar}_bin")),
        edges=bins,
        orientation="horizontal",
        color=color,
    )


for ax, title in [(ax_left, "all clouds"), (ax_right, "cloud base below 1500 m")]:
    ax.axhline(273.15, color="k", ls="-", lw=0.5)
    ax.set_xlabel("normalized count")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.1)
ax_left.set_ylabel("$T$ / K")
ax_left.legend()
for ax in [ax_left, ax_right, rect_ax]:
    sns.despine(ax=ax)

fig.tight_layout()
plt.show()

fig.savefig(scratch_path + "cloud_top_map.pdf", dpi=300)


# %%
# %%

sns.set_context("talk")
fig, axes = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
for pltds, label, color in [
    (east, "east", "#960018"),
    (west, "west", "#0085db"),
    (north, "north", "#FF7982"),
]:
    top = histogram(
        pltds.cloud_top,
        dim=["label", "step"],
        bins=np.linspace(0, 15000, 50),
        block_size=1,
    )
    lowtop = histogram(
        pltds.where(pltds.cloud_base < 1500).cloud_top,
        dim=["label", "step"],
        bins=np.linspace(0, 15000, 50),
        block_size=1,
    )
    (top / top.sum("cloud_top_bin")).plot(
        ax=axes[0], label=label, y="cloud_top_bin", color=color
    )
    (lowtop / lowtop.sum("cloud_top_bin")).plot(
        ax=axes[1], label=label, y="cloud_top_bin", color=color
    )

axes[0].legend()
# axes[0].set_xlim(0, 15000)
axes[1].set_xlim(0, 0.1)
axes[0].set_ylabel("altitude / m")
axes[0].set_title("All clouds")
axes[1].set_title("Cloud base below 1500 m")
axes[1].set_ylabel("")
for ax in axes:
    ax.set_xlabel("count")
sns.despine()
fig.tight_layout()
# fig.savefig("/scratch/m/m301046/cloud_top_hist.pdf")


# %%


sns.set_context("talk")
fig, axes = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
for pltds, label, color in [
    (north, "north", "#FF7982"),
    (west, "west", "#0085db"),
    (east, "east", "#960018"),
]:
    top = histogram(
        pltds[ctvar],
        dim=["label", "step"],
        bins=bins,
        block_size=1,
    )
    lowtop = histogram(
        pltds.where(pltds.cloud_base < 1500)[ctvar],
        dim=["label", "step"],
        bins=bins,
        block_size=1,
    )
    axes[0].stairs(
        (top / top.sum(f"{ctvar}_bin")),
        edges=bins,
        orientation="horizontal",
        color=color,
        label=label,
    )
    axes[1].stairs(
        (lowtop / lowtop.sum(f"{ctvar}_bin")),
        edges=bins,
        orientation="horizontal",
        color=color,
    )


axes[0].legend()
# axes[0].set_xlim(0, 15000)
axes[1].set_xlim(0, 0.05)
# axes[0].set_ylabel("altitude / m")
axes[0].set_title("All clouds")
axes[1].set_title("Cloud base below 1500 m")
axes[1].set_ylabel("")

for ax in axes:
    ax.set_xlabel("count")
    ax.axhline(273.15, color="k", ls="-", lw=0.5)
    ax.set_xlabel("normalized count")
axes[0].invert_yaxis()
axes[0].set_ylabel("$T$ / K")

sns.despine()
fig.tight_layout()
#
# fig.savefig("/Users/helene/Documents/scratch/cloud_top_hist.pdf")
# %%

all_radar = histogram(
    radarprops.cloud_top,
    dim=["label", "step"],
    bins=altbins,
    block_size=1,
)

sns.set_palette("Paired")
fig, axes = plt.subplots(ncols=2, sharey=True, width_ratios=[1.5, 1])
all_radar.plot(y="cloud_top_bin", ax=axes[0], label="all radar")
hists["radar"].plot(y="cloud_top_bin", ax=axes[0], label="max cloud top")

(hists["radar"] / all_radar).plot(
    y="cloud_top_bin", ax=axes[1], label="ratio", color="k"
)

axes[0].legend()
axes[0].set_xlabel("count")

axes[0].set_ylabel("cloud top / km")
axes[1].set_ylabel("")
axes[1].set_xlabel("ratio")
sns.despine()


# %%
spectimes = np.unique(specmacs.time)
specradar = (
    maxct["radar"]
    .reindex(time_idx=spectimes, method="nearest", tolerance=np.timedelta64(1, "s"))
    .dropna("time_idx")
)
specwales = (
    maxct["wales"]
    .reindex(time_idx=spectimes, method="nearest", tolerance=np.timedelta64(1, "s"))
    .dropna("time_idx")
)
# %%

t = specmacs.indexes["time"]
m = specradar.indexes["time_idx"]
start = t.searchsorted(m, side="left")
stop = t.searchsorted(m, side="right")

new_block = np.r_[True, start[1:] != stop[:-1]]
bstart = start[new_block]
bstop = stop[np.r_[new_block[1:], True]]

specsel = xr.concat(
    [specmacs.isel(time=slice(a, b)) for a, b in zip(bstart, bstop)], dim="time"
)

# %%
sns.set_palette("Paired")
fig, ax = plt.subplots()


for pltds, label in [
    (specradar, "reduced radar"),
    (maxct["radar"], "full radar"),
    (specsel, "reduced specmacs"),
    (specmacs, "full specmacs"),
    (specwales, "reduced wales"),
    (maxct["wales"], "full wales"),
]:
    ax.hist(
        pltds.values / 1000,
        bins=np.linspace(0, 15, 50),
        density=True,
        histtype="step",
        orientation="horizontal",
        label=label,
    )
ax.legend()
sns.despine()
# %%
velox = xr.open_dataset(
    "ipfs://bafybeihwsmivzu2lb6u7mzb64ux2egrtob66aklbvocvqr7nrn3apsfeo4", engine="zarr"
)

# %%

fig, axes = plt.subplots(nrows=3, figsize=(24, 12))

im = rdata.sel(
    time=slice("2024-09-03T15:10:00", "2024-09-03T15:20:00")
).reflectivity.plot(y="altitude", add_colorbar=False, ax=axes[0], rasterized=True)

# fig.colorbar(im,ax=axes[0], label="reflectivity / dBZ")
im = wales.sel(time=slice("2024-09-03T15:10:00", "2024-09-03T15:20:00")).bsrgl.plot(
    y="altitude", add_colorbar=False, ax=axes[1], rasterized=True, vmin=0, vmax=200
)

# fig.colorbar(im,ax=axes[1], extend="max")

axes[2].plot(
    velox.sel(time=slice("2024-09-03T15:10:00", "2024-09-03T15:20:00")).time,
    velox.sel(time=slice("2024-09-03T15:10:00", "2024-09-03T15:20:00")).BT_center,
    label="VELOX center",
)
axes[2].set_ylabel("VELOX center BT / K")
axes[2].set_xlim(
    np.datetime64("2024-09-03T15:10:00"), np.datetime64("2024-09-03T15:20:00")
)
axes[2].set_xlabel("time")
axes[2].axhline(0, color="k", ls="-", lw=0.5)

for ax in axes[:-1]:
    ax.set_ylabel("altitude / m")
    ax.set_ylim(3000, 10000)

fig.tight_layout()
sns.despine()
