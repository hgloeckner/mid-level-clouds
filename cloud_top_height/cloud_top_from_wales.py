# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean as cmo
import sys

sys.path.append("../")
from myutils import open_datasets
from myutils import physics_helper as physics
from myutils.constants_and_values import ml_sondes

# %%
wv, no_wv = open_datasets.open_wales(masked=True)
wv = wv.assign(q=physics.wv2q(wv))

cid = open_datasets.get_cid()
dropsondes = open_datasets.open_dropsondes(
    f"{cid}/dropsondes/Level_3/PERCUSION_Level_3.zarr"
)
# %%
flight_date = "2024-08-11"

wales_example = no_wv.sel(time=flight_date)
ml_min = 4000
ml_max = 7000


# %%
def find_highest_cloud_altitude(
    ds, new_var_name="cloud_top", variable_name="bsrgl", threshold=20
):
    ds = ds.sortby("altitude").chunk({"altitude": -1, "time": 1000})
    mask = ds[variable_name] >= threshold
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


def get_ml_cloud(wales):
    mid_level = wales.sel(altitude=slice(ml_min, ml_max))

    return (
        wales.where((mid_level.bsrgl.max(dim="altitude") >= 20))
        .compute()
        .dropna("time", how="all")
    )


def get_clear_sky(wales):
    return wales.where(
        (wales.bsrgl.max(dim="altitude") < 20) & (wales.bsrgl.count(dim="altitude") > 0)
    )


# %%
ml_cloud = get_ml_cloud(no_wv).compute().dropna("time", how="all").chunk(time=2000)
# %%
cs_wales = get_clear_sky(no_wv).compute().dropna("time", how="all").chunk(time=2000)
# %%
temps = []
for i in range(ml_cloud.sizes["time"] // 2000):
    temp_ds = ml_cloud.isel(time=slice(i * 2000, (i + 1) * 2000))
    temps.append(
        xr.merge(
            [
                temp_ds,
                find_highest_cloud_altitude(
                    temp_ds,
                    new_var_name="cloud_top",
                    variable_name="bsrgl",
                    threshold=20,
                ),
                find_highest_cloud_altitude(
                    temp_ds.sel(altitude=slice(ml_min, ml_max)),
                    new_var_name="ml_cloud_top",
                    variable_name="bsrgl",
                    threshold=20,
                ),
            ]
        )
    )

# %%
ml_cloud_with_top = xr.concat(temps, dim="time")
# %%
cmap = cmo.tools.crop_by_percent(cmo.cm.tarn_r, 30, which="both")
fig, axes = plt.subplots(
    ncols=2,
    nrows=2,
    height_ratios=(0.3, 2),
    width_ratios=(2, 0.3),
    sharex="col",
    sharey="row",
    figsize=(6, 6),
    layout="constrained",
)

ax = axes[0, 0]
hist = sns.histplot(
    ml_cloud_with_top.where(
        ml_cloud_with_top.longitude <= -40
    ).ml_cloud_top_ta.to_dataframe()["ml_cloud_top_ta"],
    ax=ax,
    color=cmap(0.2),
    stat="density",
    element="step",
    kde=True,
    fill=True,
)
hist = sns.histplot(
    ml_cloud_with_top.where(
        ml_cloud_with_top.longitude > -40
    ).ml_cloud_top_ta.to_dataframe()["ml_cloud_top_ta"],
    ax=ax,
    color=cmap(0.8),
    stat="density",
    element="step",
    kde=True,
    fill=True,
)

ax.axvline(273.15, color="grey")
ax = axes[1, 1]

hist = sns.histplot(
    ml_cloud_with_top.where(
        ml_cloud_with_top.longitude <= -40
    ).cloud_top_ta.to_dataframe(),  # ["cloud_top_ta"],
    ax=ax,
    color=cmap(0.2),
    stat="density",
    y="cloud_top_ta",
    element="step",
    kde=True,
    fill=True,
)


hist = sns.histplot(
    ml_cloud_with_top.where(
        ml_cloud_with_top.longitude > -40
    ).cloud_top_ta.to_dataframe(),
    ax=ax,
    color=cmap(0.8),
    y="cloud_top_ta",
    stat="density",
    element="step",
    kde=True,
    fill=True,
)

ax.axhline(273.15, color="grey")


ax = axes[1, 0]

p = ax.scatter(
    ml_cloud_with_top.ml_cloud_top_ta,
    ml_cloud_with_top.cloud_top_ta,
    c=ml_cloud_with_top.longitude,
    cmap=cmap,
    vmin=-65,
    vmax=-15,
    rasterized=True,
    s=3,
)

ax.axhline(273.15, color="grey")
ax.axvline(273.15, color="grey")
for ax in axes.flatten():
    ax.set_xlabel("")
    ax.set_ylabel("")
axes[1, 0].set_xlabel("mid level cloud top temperature / K")
axes[1, 0].set_ylabel("highest cloud top temperature / K")

sns.despine(ax=axes[1, 0], offset=10)
sns.despine(ax=axes[0, 0], offset={"left": 10})
sns.despine(ax=axes[1, 1], offset={"bottom": 10})

cbaxes = axes[1, 0].inset_axes((0.05, 0.95, 0.5, 0.03))
cbar = fig.colorbar(p, cax=cbaxes, orientation="horizontal")
cbar.set_label("longitude / degrees_east", size=8)
cbar.ax.tick_params(labelsize=8)
axes[0, 1].set_axis_off()
fig.savefig(
    "../plots/wales_mid_level_cloud_top.pdf",
    dpi=300,
    bbox_inches="tight",
)
# %%

print(
    ml_cloud_with_top.where(ml_cloud_with_top.longitude <= -40)
    .ml_cloud_top_ta.mean("time")
    .values
)
print(
    ml_cloud_with_top.where(ml_cloud_with_top.longitude > -40)
    .ml_cloud_top_ta.mean("time")
    .values
)
# %% air temperature difference between dropsondes and wales

sondes = dropsondes.swap_dims({"sonde_id": "launch_time"}).sortby("launch_time")
diff = (
    no_wv.sel(time=dropsondes.launch_time.values, method="nearest")
    .sortby("time")
    .assign(time=dropsondes.launch_time.values)
    .rename(time="launch_time")
    .ta
    - dropsondes.swap_dims({"sonde_id": "launch_time"}).ta
)

# %%
bsrgl = no_wv.sel(time=dropsondes.launch_time.values, method="nearest").assign(
    sonde_id=dropsondes.sonde_id.values
)
cs_sondes = dropsondes.where(
    ((bsrgl.bsrgl.max(dim="altitude") < 20) & (bsrgl.bsrgl.count(dim="altitude") > 0))
    .swap_dims({"time": "sonde_id"})
    .compute(),
    drop=True,
)

# %%

# %%
east_mid = diff.where(diff.longitude > -40).sel(altitude=slice(4000, 8000))
west_mid = diff.where(diff.longitude < -40).sel(altitude=slice(4000, 8000))
west_mid.mean("launch_time").plot(label="west")
east_mid.mean("launch_time").plot(label="east")
plt.fill_between(
    east_mid.altitude.sel(altitude=slice(4000, 8000)).values,
    east_mid.mean("launch_time").values - east_mid.std("launch_time").values,
    east_mid.mean("launch_time").values + east_mid.std("launch_time").values,
    alpha=0.2,
)
plt.fill_between(
    west_mid.altitude.sel(altitude=slice(4000, 8000)).values,
    west_mid.mean("launch_time").values - west_mid.std("launch_time").values,
    west_mid.mean("launch_time").values + west_mid.std("launch_time").values,
    alpha=0.2,
)


plt.legend()
# %% sondes with ml clouds
ml_only = ml_cloud_with_top.where(
    ml_cloud_with_top.ml_cloud_top == ml_cloud_with_top.cloud_top, drop=True
)
wales_sondes = no_wv.sel(time=dropsondes.launch_time.values, method="nearest")
mlcloud_only_sondes = dropsondes.swap_dims({"sonde_id": "launch_time"}).sel(
    launch_time=list(set(wales_sondes.time.values).intersection(ml_only.time.values)),
    method="nearest",
)

# %% get sondes cloud top temperature
sonde_ta = []
wales_ta = []
mlcloud_sondes = dropsondes.sel(sonde_id=ml_sondes)
for sonde_id in mlcloud_sondes.sonde_id.values:
    ml_top = ml_cloud_with_top.sel(
        time=mlcloud_sondes.launch_time.sel(sonde_id=sonde_id), method="nearest"
    ).ml_cloud_top.values
    ta_top = (
        mlcloud_sondes.sel(sonde_id=sonde_id)
        .ta.interpolate_na(dim="altitude")
        .sel(altitude=ml_top, method="nearest")
    )
    if not np.isnan(ta_top):
        sonde_ta.append(ta_top)
        wales_ta.append(
            ml_cloud_with_top.sel(
                time=mlcloud_sondes.launch_time.sel(sonde_id=sonde_id), method="nearest"
            ).ml_cloud_top_ta
        )
# %%
sonde_ml_ta = xr.concat(sonde_ta, dim="sonde_id")
wales_ml_ta = xr.concat(wales_ta, dim="sonde_id")
total = xr.merge(
    [
        sonde_ml_ta.rename("sonde_ta"),
        wales_ml_ta.rename("wales_ta"),
    ]
)
# %%
fig, ax = plt.subplots()
total.plot.scatter(
    wales_ta,
    sonde_ta,
    hue=total.where(total.launch_lon > -40).launch_lon,
    ax=ax,
    marker="o",
    linestyle="None",
)
ax.plot(
    [260, 280],
    [260, 280],
    color="k",
    linestyle="--",
    label="1:1 line",
)
# %%
