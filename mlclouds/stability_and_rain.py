# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import cmocean.cm as cmo
from matplotlib.colors import TwoSlopeNorm

sys.path.append("../")
from myutils import open_datasets
from myutils import physics_helper as physics
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
# %%
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

ct_wales = xr.DataArray(
    data=dh.find_highest_cloud_altitude(no_wv),
    dims="time",
    name="cloud-top",
    coords=dict(time=no_wv.time),
)


# %%
mlct = ct_wales.where((ct_wales > 4000) & (ct_wales < 8500), drop=True)
lct = ct_wales.where((ct_wales < 4000), drop=True)
hct = ct_wales.where((ct_wales >= 8500), drop=True)
# %%
mlr = (
    rdata.reindex(time=mlct.time, tolerance=np.timedelta64(1, "m"), method="nearest")
    .dropna(dim="time", how="all")
    .load()
)
rll = (
    rdata.reindex(time=lct.time, tolerance=np.timedelta64(1, "m"), method="nearest")
    .dropna(dim="time", how="all")
    .load()
)
rhl = (
    rdata.reindex(time=hct.time, tolerance=np.timedelta64(1, "m"), method="nearest")
    .dropna(dim="time", how="all")
    .load()
)
# %%
sns.set_context("paper")
fig, ax = plt.subplots(figsize=(6, 4))

histkwargs = {
    "stat": "density",
    "bins": 50,
    "binrange": (-10, 7.5),
    "element": "step",
    "fill": False,
    "linewidth": 1,
    "kde": False,
    "ax": ax,
}
sns.histplot(
    mlr.radar_doppler_velocity_corrected_no_wind.sel(altitude=slice(200, 2000))
    .median(dim="altitude")
    .to_dataframe(),
    label="ML Doppler Velocity median 200-2000m",
    palette=["blue"],
    **histkwargs,
)
sns.histplot(
    rll.radar_doppler_velocity_corrected_no_wind.sel(altitude=slice(200, 2000))
    .median(dim="altitude")
    .to_dataframe(),
    label="low/ cs Doppler Velocity  median 200-2000m",
    palette=["orange"],
    **histkwargs,
)

sns.histplot(
    rhl.radar_doppler_velocity_corrected_no_wind.sel(altitude=slice(200, 2000))
    .median(dim="altitude")
    .to_dataframe(),
    label="high Doppler Velocity  median 200-2000m",
    palette=["green"],
    **histkwargs,
)
ax.legend()
ax.axvline(-1, color="k", linestyle="--")

# %%
mlr = mlr.assign(
    meddv=mlr.radar_doppler_velocity_corrected_no_wind.sel(
        altitude=slice(200, 2000)
    ).median(dim="altitude")
)
rll = rll.assign(
    meddv=rll.radar_doppler_velocity_corrected_no_wind.sel(
        altitude=slice(200, 2000)
    ).median(dim="altitude")
)
rhl = rhl.assign(
    meddv=rhl.radar_doppler_velocity_corrected_no_wind.sel(
        altitude=slice(200, 2000)
    ).median(dim="altitude")
)
# %%


fig, axes = plt.subplots(figsize=(12, 8), nrows=3, sharey=True)

cmap = cmo.delta
norm = TwoSlopeNorm(vcenter=-1, vmax=5, vmin=-6)
mlr.radar_doppler_velocity_corrected_no_wind.sel(altitude=slice(200, None)).swap_dims(
    {"time": "meddv"}
).sortby("meddv").plot(y="altitude", norm=norm, cmap=cmap, ax=axes[0])

rll.radar_doppler_velocity_corrected_no_wind.sel(altitude=slice(200, None)).swap_dims(
    {"time": "meddv"}
).sortby("meddv").plot(y="altitude", norm=norm, cmap=cmap, ax=axes[1])
rhl.radar_doppler_velocity_corrected_no_wind.sel(altitude=slice(200, None)).swap_dims(
    {"time": "meddv"}
).sortby("meddv").plot(y="altitude", norm=norm, cmap=cmap, ax=axes[2])
