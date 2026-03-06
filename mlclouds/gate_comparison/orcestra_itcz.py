# %%
import xarray as xr
import myutils.data_helper as dh
import myutils.open_datasets as od

cids = od.get_cids()
datasets = {
    "rapsodi": od.open_radiosondes(cids["radiosondes"]),
    "beach": od.open_dropsondes(cids["dropsondes"]),
    "gate": od.open_gate(cids["gate"]),
}

for name, ds in datasets.items():
    datasets[name] = (
        ds.pipe(dh.interpolate_gaps).pipe(dh.extrapolate_sfc)
        # .pipe(pp.sel_itcz)
    )

iwv_field = xr.open_dataset("/Users/helene/Documents/Data/iwv_field.nc")
# %%
selbeach = {}
selrapsodi = {}
regions = {
    "east": dh.east,
    "west": dh.west,
}
for name, region in regions.items():
    ds = dh.sel_sub_domain(datasets["beach"], region, item_var="sonde")

    selbeach[name] = iwv_field.sel(
        time=ds.launch_time,
        latitude=ds.launch_lat,
        longitude=ds.launch_lon % 360,
        method="nearest",
    )
    ds = dh.sel_sub_domain(datasets["rapsodi"], region, item_var="sonde")

    selrapsodi[name] = iwv_field.sel(
        time=ds.launch_time,
        latitude=ds.launch_lat,
        longitude=ds.launch_lon % 360,
        method="nearest",
    )

# %%
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(figsize=(12, 4), ncols=2)

for idx, (name, binrange) in enumerate(zip(["east", "west"], [(-6, 3), (-6, 18)])):
    sns.histplot(
        data=selbeach[name].distance,
        bins=50,
        binrange=binrange,
        color="C0",
        alpha=0.5,
        element="step",
        stat="percent",
        label="beach",
        ax=axes[idx],
    )
    sns.histplot(
        data=selrapsodi[name].distance,
        bins=50,
        binrange=binrange,
        color="C1",
        alpha=0.5,
        stat="percent",
        label="rapsodi",
        element="step",
        ax=axes[idx],
    )
    axes[idx].set_title(name)

    axes[idx].axvline(selbeach[name].distance.median(), label="beach", color="C0")
    axes[idx].axvline(selrapsodi[name].distance.median(), label="rapsodi", color="C1")

    print(f"beach median distance {name}", selbeach[name].distance.median().values)
    print(f"rapsodi median distance {name}", selrapsodi[name].distance.median().values)
axes[0].legend()
sns.despine(offset={"left": 10})

fig.savefig(
    "../../plots/iwv_distance_histogram.pdf",
    bbox_inches="tight",
)
# %%

# %%
import cartopy.crs as ccrs

fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={"projection": ccrs.Robinson()})

iwv_field.mean("time").distance.plot(
    transform=ccrs.PlateCarree(),
)
ds = datasets["beach"].swap_dims({"sonde": "launch_time"}).sel(launch_time="2024-08-11")
ax.plot(
    ds.launch_lon,
    ds.launch_lat,
    "o",
    transform=ccrs.PlateCarree(),
    label="beach",
)

dtest = iwv_field.sel(
    time=ds.launch_time,
    latitude=ds.launch_lat,
    longitude=ds.launch_lon % 360,
    method="nearest",
)
ax.plot(
    dtest.longitude,
    dtest.latitude,
    "x",
    transform=ccrs.PlateCarree(),
    label="nearest gridpoint",
)

# %%

import cartopy.crs as ccrs

r = dh.sel_sub_domain(
    datasets["beach"],
    dh.east,
    item_var="sonde",
)

fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={"projection": ccrs.Robinson()})


p = ax.scatter(
    r.launch_lon,
    r.launch_lat,
    c=r.sel(altitude=5820).ta,
    transform=ccrs.PlateCarree(),
    label="beach",
)
plt.colorbar(p, ax=ax, label="T at 5460 m / K")
