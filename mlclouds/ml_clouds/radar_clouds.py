# %%
import pandas as pd

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage as ndimage
from xhistogram.xarray import histogram


# %%
rdata = xr.open_dataset(
    "ipfs://bafybeifxtmq5mpn7vwiiwl4vlpoil7rgm2tnhmkeyqsyudleqegxzvwl3a",
    engine="zarr",
    chunks={"altitude": -1},
)

rdata = (
    rdata.assign(
        reflectivity=(10 * np.log10(rdata.radar_reflectivity)),
        snr_db=10 * np.log10(rdata.radar_signal_to_noise),
    )
    .sel(time=slice(np.datetime64("2024-08-11"), np.datetime64("2024-09-29")))
    .sortby("altitude")
    .sel(altitude=slice(200, None))
)
rdata = rdata.where(rdata.snr_db > -17)


# %%
def get_labels_by_segment(rdata, max_gap):
    labels = np.full((rdata.sizes["time"], rdata.sizes["altitude"]), np.nan)
    structure = ndimage.generate_binary_structure(2, 2)
    gaps = np.diff(rdata.time) > max_gap
    segments = np.concatenate(([0], np.cumsum(gaps)))
    offset = 0
    for id in np.unique(segments):
        seg_data = rdata.isel(time=np.where(segments == id)[0])
        lab, n = ndimage.label(
            seg_data.reflectivity.interpolate_na("altitude", max_gap=90).notnull(),
            structure=structure,
        )
        lab = np.where(lab > 0, lab + offset, np.nan)
        idxs = np.where(segments == id)[0]
        labels[idxs, :] = lab
        offset = offset + n
    return labels


seg = get_labels_by_segment(rdata, np.timedelta64(5, "m"))


# %%
def mask_small_objects(seg, min_size):
    sizes = np.bincount(np.nan_to_num(seg).astype(int).ravel())
    valid_id = np.flatnonzero(sizes > min_size)
    return np.where(np.isin(seg, valid_id), seg, np.nan)


sizes = mask_small_objects(seg, 3)
# %%
ti, ai = np.nonzero(~np.isnan(seg))
df = pd.DataFrame(
    {
        "label": seg[ti, ai].astype(int),
        "ti": ti,
        "altitude": rdata.altitude.values[ai],
    }
)

stats = (
    df.groupby(["label", "ti"])["altitude"]
    .agg(cloud_base="min", cloud_top="max")
    .reset_index()
)
stats["step"] = stats["ti"] - stats.groupby("label")["ti"].transform("min")
stats["time_idx"] = rdata["time"].values[stats["ti"]]

ds = stats.set_index(["label", "step"])[
    ["cloud_top", "cloud_base", "time_idx"]
].to_xarray()

ds = ds.assign(
    cloud_depth=ds.cloud_top - ds.cloud_base,
)
# %%
ds.to_zarr("/scratch/m/m301046/cloud_props_radar.zarr", mode="w")

# %%
ds = xr.open_dataset("/scratch/m/m301046/cloud_props_radar.zarr")

# %%
ds = ds.assign(mean_time=ds.time_idx.groupby("label").mean("step"))

# %%
east = ds.where(ds.mean_time < np.datetime64("2024-09-05"), drop=True)
west = ds.where(ds.mean_time > np.datetime64("2024-09-07"), drop=True)

# %%


# %%

sns.set_context("talk")
fig, axes = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
for ds, label, color in [(east, "east", "#960018"), (west, "west", "#0085db")]:
    top = histogram(
        ds.cloud_top,
        dim=["label", "step"],
        bins=np.linspace(0, 15000, 50),
        block_size=1,
    )
    lowtop = histogram(
        ds.where(ds.cloud_base < 1500).cloud_top,
        dim=["label", "step"],
        bins=np.linspace(0, 15000, 50),
        block_size=1,
    )
    top.plot(ax=axes[0], label=label, y="cloud_top_bin", color=color)
    lowtop.plot(ax=axes[1], label=label, y="cloud_top_bin", color=color)

axes[0].legend()
axes[0].set_xlim(0, 15000)
axes[1].set_xlim(0, 4000)
axes[0].set_ylabel("altitude / m")
axes[0].set_title("All clouds")
axes[1].set_title("Cloud base below 1500 m")
axes[1].set_ylabel("")
for ax in axes:
    ax.set_xlabel("count")
sns.despine()
fig.tight_layout()
fig.savefig("/scratch/m/m301046/cloud_top_hist.pdf")
