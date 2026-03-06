# %%

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import numpy as np
from xhistogram.xarray import histogram
import cmocean as cmo
import myutils.data_helper as dh
import radiation_for_sondes.rrtmg.angles as angles
import myutils.open_datasets as od
import pandas as pd
import scipy.signal as signal

regions = {
    "west": dh.west,
    "east": dh.east,
    "north": dh.north,
}
# %%
"""
radar_ship = xr.open_dataset("ipfs://bafybeiakgiqypaykbbyxxaesqxz2g4kjwjactcpvju2tjutztm5hpxl2km", engine="zarr")

wv, no_wv = od.open_wales(masked=False, local=True)
no_wv = no_wv.sel(time=slice(np.datetime64("2024-08-10"), np.datetime64("2024-09-29")))
no_wv = xr.where(no_wv.bsrgl_flags == 8, 100, no_wv)
no_wv = no_wv.where((no_wv.bsrgl_flags == 0) | (no_wv.bsrgl_flags == 8)).sortby(
    "altitude"
)
no_wv = no_wv.sel(altitude=slice(0, None))



radar_halo = xr.open_dataset(
    "ipfs://bafybeifxtmq5mpn7vwiiwl4vlpoil7rgm2tnhmkeyqsyudleqegxzvwl3a", engine="zarr"
)

radar_halo = (
    radar_halo.assign(reflectivity=(10 * np.log(radar_halo.radar_reflectivity)))
    .sel(time=slice(np.datetime64("2024-08-10"), np.datetime64("2024-09-29")))
    .sortby("altitude")
)
rdata = xr.DataArray(
    data=dh.find_highest_cloud_altitude(
        radar_halo.sel(altitude=slice(200, None)),
        variable_name="reflectivity",
        threshold=-1e3,
    ),
    dims="time",
    name="cloud-top",
)
rdata = rdata.assign_coords(
    time=radar_halo.time,
    lat=radar_halo.latitude,
    lon=radar_halo.longitude,
    mu0=("time", xr.apply_ufunc(
        angles.cos_zenith_angle,
        radar_halo.time,
        radar_halo.latitude,
        radar_halo.longitude,
        vectorize=True,
        dask="parallelized",
    ).values, {"long_name": "cosine of solar zenith angle at launch time", "units": "1"}),
    )#.to_dataframe()


wales = xr.DataArray(
        data=dh.find_highest_cloud_altitude(no_wv.sel(altitude=slice(200, None))),
        dims="time",
        name="cloud-top",
    )#.to_dataframe()

wales = wales.assign_coords(
    time=no_wv.time,
     mu0=("time", xr.apply_ufunc(
        angles.cos_zenith_angle,
        no_wv.time,
        no_wv.latitude,
        no_wv.longitude,
        vectorize=True,
        dask="parallelized",
    ).values, {"long_name": "cosine of solar zenith angle at launch time", "units": "1"}),

).rename({"latitude":"lat", "longitude":"lon"})


radar= radar_ship[["cloud_top_height_agl","longitude", "latitude"]].rename(
        {"cloud_top_height_agl": "cloud-top", "longitude": "lon", "latitude": "lat"}
    )
radar = radar.dropna(dim="time", subset=["lat", "lon", "time"])
radar = radar.assign_coords(
    mu0=("time", xr.apply_ufunc(
        angles.cos_zenith_angle,
        radar.time,
        radar.lat,
        radar.lon,
        vectorize=True,
        dask="parallelized",
    ).values, {"long_name": "cosine of solar zenith angle at launch time", "units": "1"}),
)

od.write_ds(wales.chunk(time=-1).to_dataset(), f"{folder}wales_ct.zarr")
od.write_ds(radar.chunk(time=-1), f"{folder}radar_ship.zarr")
od.write_ds(rdata.chunk(time=-1).to_dataset(), f"{folder}radar_halo.zarr")
"""
# %%
folder = "/scratch/m/m301046/mlclouds/"
folder = "/Users/helene/Documents/Data/mlclouds/"
# %%
lim = 2.5e-12
shiplim = 1.75e-12
wales = xr.open_dataset(f"{folder}wales_ct.zarr")
wales = wales.assign(
    hour=(
        ("time"),
        pd.to_datetime(
            angles.get_local_time(wales.time.values, wales.lon.values)
        ).hour.values,
    )
)
shipradar = xr.open_dataset(f"{folder}radar_ship.zarr")
shipradar = shipradar.assign(
    hour=(
        ("time"),
        pd.to_datetime(
            angles.get_local_time(shipradar.time.values, shipradar.lon.values)
        ).hour.values,
    )
)

haloradar = xr.open_dataset(f"{folder}radar_halo.zarr")
haloradar = haloradar.assign(
    hour=(
        ("time"),
        pd.to_datetime(
            angles.get_local_time(haloradar.time.values, haloradar.lon.values)
        ).hour.values,
    )
)

# haloradar = haloradar.reset_coords().resample(time="30s").nearest(tolerance="15s")

# %%
peak_time = []
sunset = {}
sunrise = {}
for name, data in zip(["wales", "haloradar"], [wales, haloradar]):
    suns = []
    sunr = []
    for day in pd.date_range("2024-08-11", "2024-09-29"):
        try:
            ds = data.sel(time=str(day).split(" ")[0])
        except KeyError:
            continue
        else:
            peak = signal.find_peaks(ds.mu0, distance=25000)[0]
            try:
                peak_time.append(ds.isel(time=peak).time.values[0])
            except IndexError:
                continue
            else:
                sunr.append(ds.sel(time=slice(None, peak_time[-1])))
                suns.append(ds.sel(time=slice(peak_time[-1], None)))

    sunrise[name] = xr.concat(sunr, dim="time")
    sunrise[name] = sunrise[name]
    sunset[name] = xr.concat(suns, dim="time")
    sunset[name] = sunset[name]
# %%
peak = signal.find_peaks(shipradar.mu0, distance=2000)[0]
valleys = signal.find_peaks(-shipradar.mu0, distance=2000)[0]
rises = [shipradar.isel(time=slice(valleys[i], peak[i])) for i in range(len(peak))]
sets = [
    shipradar.isel(time=slice(peak[i], valleys[i + 1])) for i in range(len(peak) - 1)
]
sunrise["shipradar"] = xr.concat(rises, dim="time")
sunrise["shipradar"] = sunrise["shipradar"]
sunset["shipradar"] = xr.concat(sets, dim="time")
sunset["shipradar"] = sunset["shipradar"]
# %%
sns.set_palette("tab10")
name = "shipradar"

for ds, c in zip([sunset[name], sunrise[name]], ["C0", "C1"]):
    plt.scatter(pd.to_datetime(ds.time).hour, ds.mu0, c=c, linestyle="", marker="o")

# %%
region = "east"
kwargs = {
    "vmax": 0.008,
    "cmap": "cmo.ice",
    "add_colorbar": False,
}
nmu0 = 12

mu0bins = angles.get_mu_day(np.datetime64("2024-06-21T00:00:00"), lat=0, lon=0, lim=13)
mu0bins = np.sort(np.cos(np.linspace(0, np.pi, 12)))  # %%

datasets = {
    "shipradar": shipradar,
    "haloradar": haloradar,
    "wales": wales,
}

name = "shipradar"
binvar = "cloud-top"
bins = np.array([0, 4000, 8000, 20000])

cm = 1 / 2.54  # centimeters to inches
cw = 20 * cm  # figure width in inches
fig, axes = plt.subplots(
    ncols=2,
    nrows=2,
    width_ratios=[1, 0.02],
    height_ratios=[0.2, 0.8],
    figsize=(cw, 0.6 * cw),
    sharex="col",
)


axes[0, 1].set_visible(False)


rise_ds = dh.sel_sub_domain(
    sunrise[name],
    regions[region],
    item_var="time",
    lat_var="lat",
    lon_var="lon",
)
set_ds = dh.sel_sub_domain(
    sunset[name],
    regions[region],
    item_var="time",
    lat_var="lat",
    lon_var="lon",
)

hist_rise = histogram(
    rise_ds["mu0"],
    rise_ds["cloud-top"],
    bins=[mu0bins, np.linspace(0, 12000, 100)],
)
hist_set = histogram(
    set_ds["mu0"],
    set_ds["cloud-top"],
    bins=[mu0bins, np.linspace(0, 12000, 100)],
)
hist_set = hist_set.assign_coords(mu0_bin=2 - hist_set.mu0_bin)

hist = xr.concat([hist_rise, hist_set], dim="mu0_bin").sortby("mu0_bin")
hist_mu_rise = histogram(
    sunrise[name]["mu0"],
    bins=[mu0bins],
)
hist_mu_set = histogram(
    sunset[name]["mu0"],
    bins=[mu0bins],
)
hist_mu_set = hist_mu_set.assign_coords(mu0_bin=2 - hist_mu_set.mu0_bin)

normalize = xr.concat([hist_mu_set, hist_mu_rise], "mu0_bin").sortby(
    "mu0_bin"
)  # hist.sum()


p = (hist_rise / normalize).plot(ax=axes[1, 0], y="cloud-top_bin", **kwargs)
(hist_set / normalize).plot(ax=axes[1, 0], y="cloud-top_bin", x="mu0_bin", **kwargs)

fig.colorbar(
    p,
    cax=axes[1, 1],
    pad=0.01,
    extend="max",
)

axes[1, 0].set_xlabel("cos zenith angle (separate sunrise/sunset)")
axes[1, 0].set_xticks(
    [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3],
    labels=[
        -1,
        -0.5,
        "0 \n sunrise",
        0.5,
        "1 \n solar noon",
        0.5,
        "0 \n sunset",
        -0.5,
        -1,
    ],
)
# axes[1, 0].set_xticks([0, 0.5, 1, 1.5, 2], labels=["0 \n sunrise", 0.5, "1 \n solar noon", 0.5, "0 \n sunset"                                                   ])
axes[1, 0].set_ylabel("Cloud top height / m")


colors = {
    "4-8 km": "#B8892E",
    "< 4 km": "#4F6A82",
    "> 8 km": "#C9C3E6",
}
for name, ls in zip(["shipradar", "haloradar", "wales"], ["-", "--", ":"]):
    rise_ds = dh.sel_sub_domain(
        sunrise[name],
        regions[region],
        item_var="time",
        lat_var="lat",
        lon_var="lon",
    )
    set_ds = dh.sel_sub_domain(
        sunset[name],
        regions[region],
        item_var="time",
        lat_var="lat",
        lon_var="lon",
    )

    hist_mean_set = histogram(
        set_ds["mu0"],
        set_ds[binvar],
        bins=[mu0bins, bins],
    )
    hist_mean_set = hist_mean_set.assign_coords(mu0_bin=2 - hist_mean_set.mu0_bin)
    hist_mean_rise = histogram(
        rise_ds["mu0"],
        rise_ds[binvar],
        bins=[mu0bins, bins],
    )
    hist_mean = xr.concat([hist_mean_rise, hist_mean_set], dim="mu0_bin").sortby(
        "mu0_bin"
    )

    for clayer, label in zip([6000, 2000, 14000], ["4-8 km", "< 4 km", "> 8 km"]):
        if ls == "-":
            lname = label
        else:
            lname = None
        normed = hist_mean / hist_mean.sum(f"{binvar}_bin")
        normed.sel({f"{binvar}_bin": clayer}).plot(
            ax=axes[0, 0],
            label=lname,
            color=colors[label],
            linestyle=ls,
        )


axes[0, 0].set_xlabel("")
axes[0, 0].set_title("")
axes[0, 0].set_ylabel("density")
for x in [0, 1, 2]:
    axes[0, 0].axvline(x, color="k", linestyle="--")

sns.despine()
fig.tight_layout()
axes[0, 0].legend(bbox_to_anchor=(1.2, 0.51), loc="center right")
# fig.savefig(f"/scratch/m/m301046/mlclouds/plots/cth_vs_mu0_haloradar_norm_mu0_{region}.pdf")

# %%


fig, axes = plt.subplots(nrows=3, figsize=(cw, 0.9 * cw), sharex=True)
colors = {
    "4-8 km": "#B8892E",
    "< 4 km": "#4F6A82",
    "> 8 km": "#C9C3E6",
}
for name, ls, dataname in zip(
    ["shipradar", "haloradar", "wales"],
    ["-", "--", ":"],
    ["Meteor", "HALO Radar", "Wales"],
):
    set_ds = dh.sel_sub_domain(
        sunset[name],
        regions[region],
        item_var="time",
        lat_var="lat",
        lon_var="lon",
    )
    rise_ds = dh.sel_sub_domain(
        sunrise[name],
        regions[region],
        item_var="time",
        lat_var="lat",
        lon_var="lon",
    )

    hist_mean_set = histogram(
        set_ds["mu0"],
        set_ds[binvar],
        bins=[mu0bins, bins],
    )
    hist_mean_set = hist_mean_set.assign_coords(mu0_bin=2 - hist_mean_set.mu0_bin)
    hist_mean_rise = histogram(
        rise_ds["mu0"],
        rise_ds[binvar],
        bins=[mu0bins, bins],
    )
    hist_mean = xr.concat([hist_mean_rise, hist_mean_set], dim="mu0_bin").sortby(
        "mu0_bin"
    )

    for clayer, label, zorder in zip(
        (bins[:-1] + bins[1:]) / 2,
        [
            "< 4 km",
            "4-8 km",
            "> 8 km",
        ],
        [1, 3, 2],
    ):
        if ls == "-":
            lname = label
        else:
            lname = None

        normed = hist_mean / hist_mean.sum("mu0_bin")
        normed = normed / normed.sum(f"{binvar}_bin")
        normed.sel({f"{binvar}_bin": clayer}).plot(
            ax=axes[0],
            label=lname,
            color=colors[label],
            linestyle=ls,
            zorder=zorder,
            linewidth=zorder,
        )

        normed = hist_mean / hist_mean.sum(f"{binvar}_bin")
        normed.sel({f"{binvar}_bin": clayer}).plot(
            ax=axes[1],
            label=lname,
            color=colors[label],
            linestyle=ls,
            zorder=zorder,
            linewidth=zorder,
        )

    hist_mu_set = histogram(
        set_ds["mu0"],
        bins=mu0bins,
    )
    hist_mu_set = hist_mu_set.assign_coords(xvar=2 - hist_mu_set.mu0_bin).swap_dims(
        {"mu0_bin": "xvar"}
    )
    hist_mu_rise = histogram(
        rise_ds["mu0"],
        bins=mu0bins,
    )
    hist_mu_rise = hist_mu_rise.assign_coords(xvar=hist_mu_rise.mu0_bin).swap_dims(
        {"mu0_bin": "xvar"}
    )
    hist_mu = xr.concat([hist_mu_rise, hist_mu_set], dim="xvar").sortby("xvar")

    (hist_mu / hist_mu.sum()).plot(ax=axes[-1], color="k", ls=ls, label=dataname)


axes[1].set_title(f"normed by {binvar} column (equals 1 for each mu0)")
# axes[2].set_title("normed by mu0 column (equals 1 for each cth)")
axes[0].set_title(f"first normed by {binvar} column, then by mu0 column")
axes[0].legend()
sns.despine()
fig.tight_layout()
axes[-1].set_xticks(
    [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3],
    labels=[
        -1,
        -0.5,
        "0 \n sunrise",
        0.5,
        "1 \n solar noon",
        0.5,
        "0 \n sunset",
        -0.5,
        -1,
    ],
)
axes[-1].legend()
for ax in axes:
    ax.set_ylabel("Probability")
    for i in [0, 1, 2]:
        ax.axvline(i, color="grey", alpha=0.5, linestyle="-")
    ax.set_xlabel("")
axes[-1].set_xlabel("cos zenith angle (separate sunrise/sunset)")

# %%
binvar = "cloud-top"

bins = np.array([0, 4000, 8000, 20000])

fig, ax = plt.subplots(nrows=1, figsize=(cw, 0.3 * cw))
colors = {
    "4-8 km": "#B8892E",
    "< 4 km": "#4F6A82",
    "> 8 km": "#C9C3E6",
}
for name, ls, dataname in zip(
    ["shipradar", "haloradar", "wales"],
    ["-", "--", ":"],
    ["Meteor", "HALO Radar", "Wales"],
):
    set_ds = dh.sel_sub_domain(
        sunset[name],
        regions[region],
        item_var="time",
        lat_var="lat",
        lon_var="lon",
    )
    rise_ds = dh.sel_sub_domain(
        sunrise[name],
        regions[region],
        item_var="time",
        lat_var="lat",
        lon_var="lon",
    )
    full_ds = dh.sel_sub_domain(
        datasets[name],
        regions[region],
        item_var="time",
        lat_var="lat",
        lon_var="lon",
    )

    hist_mean_set = histogram(
        set_ds["mu0"],
        set_ds[binvar],
        bins=[mu0bins, bins],
    )
    hist_mean_set = hist_mean_set.assign_coords(mu0_bin=2 - hist_mean_set.mu0_bin)
    hist_mean_rise = histogram(
        rise_ds["mu0"],
        rise_ds[binvar],
        bins=[mu0bins, bins],
    )
    hist_mean = xr.concat([hist_mean_rise, hist_mean_set], dim="mu0_bin").sortby(
        "mu0_bin"
    )

    hist_mu_set = histogram(
        set_ds["mu0"],
        bins=mu0bins,
    )
    hist_mu_set = hist_mu_set.assign_coords(mu0_bin=2 - hist_mu_set.mu0_bin)
    hist_mu_rise = histogram(
        rise_ds["mu0"],
        bins=mu0bins,
    )
    hist_mu_rise = hist_mu_rise
    hist_mu = xr.concat([hist_mu_rise, hist_mu_set], dim="mu0_bin")

    hist_hour = histogram(
        full_ds["hour"],
        full_ds[binvar],
        bins=[np.arange(0, 25, 1), bins],
    )

    for clayer, label, zorder in zip(
        (bins[:-1] + bins[1:]) / 2,
        [
            "< 4 km",
            "4-8 km",
            "> 8 km",
        ],
        [1, 3, 2],
    ):
        if ls == "-":
            lname = label
        else:
            lname = None

        normed = hist_mean.where(
            hist_mean.sum("cloud-top_bin") > 1000
        )  # (hist_mean /hist_mean.sum("mu0_bin"))

        # anormed = normed / normed.sum("mu0_bin")
        normed = normed / hist_mu
        normed.sel({f"{binvar}_bin": clayer}).plot(
            ax=ax,
            label=lname,
            color=colors[label],
            linestyle=ls,
            zorder=zorder,
            linewidth=zorder,
        )
        """
        normed = hist_hour.where(hist_hour.sum("cloud-top_bin") > 1000)#(hist_hour /hist_hour.sum("hour_bin"))
        normed = normed / normed.sum(f"{binvar}_bin")
        normed.sel({f"{binvar}_bin": clayer}).plot(ax=axes[1], label=lname, color=colors[label], linestyle=ls, zorder=zorder, linewidth=zorder)
        """
ax.set_ylabel("Probability")
ax.set_title("")
for i in [0, 1, 2]:
    ax.axvline(i, color="grey", alpha=0.5, linestyle="-")
# axes[1].axvline(12, color="grey",alpha=0.5, linestyle="-")
# axes[1].set_xlabel("Hour of day")
ax.set_xlabel("cos zenith angle (separate sunrise/sunset)")
ax.legend()
sns.despine()
fig.tight_layout()
ax.set_xticks(
    [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3],
    labels=[
        -1,
        -0.5,
        "0 \n sunrise",
        0.5,
        "1 \n solar noon",
        0.5,
        "0 \n sunset",
        -0.5,
        -1,
    ],
)
fig.savefig(f"../../plots/cth_vs_mu0_hourly_haloradar_norm_mu0_{region}.pdf")

# %% plot with time
timebins = np.arange(0, 24.1, 1)

binvar = "mu0"
bins = mu0bins


sns.set_palette("Paired")

fig, axes = plt.subplots(nrows=2, figsize=(cw, 0.6 * cw), sharex=True)
colors = {
    "4-8 km": "#B8892E",
    "< 4 km": "#4F6A82",
    "> 8 km": "#C9C3E6",
}
for name, ls, dataname in zip(
    ["haloradar"], ["-", "--", ":"], ["Meteor", "HALO Radar", "Wales"]
):
    plt_ds = dh.sel_sub_domain(
        datasets[name],
        regions[region],
        item_var="time",
        lat_var="lat",
        lon_var="lon",
    )

    plt_ds = plt_ds.assign(
        hour=(
            ("time"),
            pd.to_datetime(
                angles.get_local_time(plt_ds.time.values, plt_ds.lon.values)
            ).hour.values,
        )
    )

    for name, ds in zip(["rise", "set"], [sunrise[name], sunset[name]]):
        (
            histogram(
                ds.hour,
                bins=[timebins],
                weights=ds[binvar],
            )
            / histogram(
                ds.hour,
                bins=[timebins],
            )
        ).plot(ax=axes[0], label=name, linestyle=ls)
        sns.histplot(
            ds.hour,
            ax=axes[1],
            label=name,
        )
axes[1].set_xticks(timebins, labels=timebins.astype(int))
sns.despine()
axes[0].legend()
# %%
binvar = "lat"
bins = np.linspace(2, 15, 15)
fig, axes = plt.subplots(nrows=2, figsize=(cw, 0.6 * cw))
colors = {
    "4-8 km": "#B8892E",
    "< 4 km": "#4F6A82",
    "> 8 km": "#C9C3E6",
}
name = "haloradar"
for i, (name, value) in enumerate(zip(["haloradar", "shipradar"], [0.03, 0.01])):
    set_ds = dh.sel_sub_domain(
        sunset[name],
        regions[region],
        item_var="time",
        lat_var="lat",
        lon_var="lon",
    )
    rise_ds = dh.sel_sub_domain(
        sunrise[name],
        regions[region],
        item_var="time",
        lat_var="lat",
        lon_var="lon",
    )
    full_ds = dh.sel_sub_domain(
        datasets[name],
        regions[region],
        item_var="time",
        lat_var="lat",
        lon_var="lon",
    )

    hist_mean_set = histogram(
        set_ds["mu0"],
        set_ds[binvar],
        bins=[mu0bins, bins],
    )
    hist_mean_set = hist_mean_set.assign_coords(mu0_bin=2 - hist_mean_set.mu0_bin)
    hist_mean_rise = histogram(
        rise_ds["mu0"],
        rise_ds[binvar],
        bins=[mu0bins, bins],
    )
    hist_mean = xr.concat([hist_mean_rise, hist_mean_set], dim="mu0_bin").sortby(
        "mu0_bin"
    )

    hist_hour = histogram(
        full_ds["hour"],
        full_ds[binvar],
        bins=[np.arange(0, 25, 1), bins],
    )
    (hist_mean / hist_mean.sum()).plot(
        ax=axes[i],
        label=name,
        add_colorbar=False,
        cmap="cmo.ice",
        y="lat_bin",
        x="mu0_bin",
        vmax=value,
        vmin=0,
    )
    # (hist_hour / hist_hour.sum()).plot(ax=axes[1], label=name, add_colorbar=False, cmap="cmo.ice", y="lat_bin", x="hour_bin", vmax=0.01, vmin=0)

for ax in axes:
    ax.set_ylabel("latitude / $^\circ$N")
    ax.set_title("")
    ax.set_xticks(
        [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3],
        labels=[
            -1,
            -0.5,
            "0 \n sunrise",
            0.5,
            "1 \n solar noon",
            0.5,
            "0 \n sunset",
            -0.5,
            -1,
        ],
    )
    ax.set_xlabel("cos zenith angle (separate sunrise/sunset)")
    for i in [0, 1, 2]:
        ax.axvline(i, color="grey", alpha=0.5, linestyle="-")
# axes[1].set_xlabel("Hour of day")
# axes[0].set_xlabel("cos zenith angle (separate sunrise/sunset)")
sns.despine()
fig.tight_layout()
fig.savefig(f"../../plots/lat_diurnal_{region}.pdf")
