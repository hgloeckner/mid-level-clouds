# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

import easygems.healpix as egh
import sys
import intake
from scipy import signal
import cartopy.crs as ccrs
import cmocean
import cartopy.feature as cf

sys.path.append("../")
import myutils.open_datasets as opends  # noqa
import myutils.plot_helper as ph  # noqa


# %%
cid = opends.get_cid()
rs = opends.open_radiosondes(f"{cid}/Radiosondes/RAPSODI_RS_ORCESTRA_level2.zarr")
rs = rs.where(rs.launch_lon > -40, drop=True)


ds = opends.open_dropsondes(f"{cid}/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr")
ds = ds.where(ds.launch_lon > -40, drop=True)

gate = opends.open_gate("QmeAFUdB3PZHRtCd441HjRGZPmEadtskXsnL34C9xigH3A")

# %%# %%
reanalysis = opends.open_reanalysis(chunks={}, zoom=7)
# %%

era = reanalysis["ERA5"]
extent = egh.get_extent_mask(era, extent=[-27, -20, 5, 12])
era = era.sel(time=era.time.dt.month.isin([8, 9])).where(extent)
# %%


def ocean(ds):
    return ds.ocean_fraction_surface == 1


cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")

icon = cat.ICON["ngc3028"](zoom=7, chunks="auto").to_dask()
sea = icon.assign_coords(cell=icon.cell).where(icon.ocean_fraction_surface > 0.9)


# %%
era2tocean = (
    era["2t"].where(~np.isnan(sea.ocean_fraction_surface)).groupby("time.year").mean()
)
merra2tocean = reanalysis["MERRA2"]["t2m"].where(~np.isnan(sea.ocean_fraction_surface))
merra2tocean = (
    merra2tocean.sel(time=merra2tocean.time.dt.month.isin([8, 9]))
    .where(extent)
    .dropna(dim="time", how="all")
    .groupby("time.year")
    .mean()
)
jra2tocean = reanalysis["JRA3Q"]["mean2t"].where(~np.isnan(sea.ocean_fraction_surface))
jra2tocean = (
    jra2tocean.sel(time=jra2tocean.time.dt.month.isin([8, 9]))
    .where(extent)
    .isel(time=slice(1, None))
    .groupby("time.year")
    .mean()
)

# %%
best = xr.open_dataset("/work/mh0066/m301046/Data/BEST/Global_TAVG_Gridded_1deg.nc")


# %%
def get_useful_times(ds):
    years = ds.time.astype(int)
    months = np.ceil((best.time - best.time.astype(int)) * 12).astype(int)

    return ds.assign(
        time=[
            np.datetime64(f"{year}-{month:02d}-01")
            for year, month in zip(years.values, months.values)
        ]
    )


best_data = get_useful_times(best).sel(latitude=slice(5, 12), longitude=slice(-27, -20))
# %%


projection = ccrs.Robinson(central_longitude=10)
fig, ax = plt.subplots(
    figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True
)
ax.set_extent([-67, -10, -2, 22])

best_data.temperature.mean(dim="time").plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
)
ax.add_feature(cf.COASTLINE, linewidth=0.8)
ax.add_feature(cf.BORDERS, linewidth=0.4)
# %%
best_data.temperature.mean(dim=["latitude", "longitude"]).plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
)
# %%

best_summer = (
    best_data.sel(time=best_data.time.dt.month.isin([8, 9])).groupby("time.year").mean()
)
# %%
detrended_era = signal.detrend(
    era2tocean.mean("cell").sel(year=slice(1974, None)).values,
)
detrended_merra = signal.detrend(
    merra2tocean.mean("cell").values,
)  # bp=[64])  # 2012
detrended_jra = signal.detrend(
    jra2tocean.mean("cell").sel(year=slice(1974, None)).values,  # bp=[45, 129]
)  # 1970, 2012
detrended_best = signal.detrend(
    best_summer.temperature.mean(dim=["latitude", "longitude"])
    .sel(year=slice(1974, None))
    .values
)

# Decompose the time series
# %%

# %%
sns.color_palette("bright")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(era2tocean.year.sel(year=slice(1974, None)), detrended_era, label="ERA5")
ax.fill_between(
    era2tocean.year.sel(year=slice(1974, None)),
    -era2tocean.mean("cell").sel(year=slice(1974, None)).std("year").values,
    era2tocean.mean("cell").sel(year=slice(1974, None)).std("year").values,
    alpha=0.1,
    color="k",
)
ax.plot(merra2tocean.year, detrended_merra, label="MERRA2")
ax.fill_between(
    merra2tocean.year,
    -merra2tocean.mean("cell").std("year").values,
    merra2tocean.mean("cell").std("year").values,
    alpha=0.1,
    color="k",
)
ax.plot(jra2tocean.sel(year=slice(1974, None)).year, detrended_jra, label="JRA3Q")
ax.fill_between(
    jra2tocean.sel(year=slice(1974, None)).year,
    -jra2tocean.mean("cell").sel(year=slice(1974, None)).std("year").values,
    jra2tocean.mean("cell").sel(year=slice(1974, None)).std("year").values,
    alpha=0.1,
    color="k",
)
ax.plot(best_summer.sel(year=slice(1974, None)).year, detrended_best, label="BEST")
ax.fill_between(
    best_summer.sel(year=slice(1974, None)).year,
    -best_summer.temperature.mean(dim=["latitude", "longitude"])
    .sel(year=slice(1974, None))
    .std("year")
    .values,
    best_summer.temperature.mean(dim=["latitude", "longitude"])
    .sel(year=slice(1974, None))
    .std("year")
    .values,
    alpha=0.1,
    color="k",
)


ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
ax.legend()
ax.set_xlabel("Year")
ax.set_ylabel("August/September mean anomaly / K")
ax.set_title("Detrended 2m/sfc air temperature for lon: [-27, -20] and lat: [5, 12]")
sns.despine(offset=10)
xticks = ax.get_xticks()
xticks = np.append(xticks, [1974, 2024])
ax.set_xticks(xticks)
ax.set_xlim(1973, 2025)
fig.savefig(
    "../plots/gate/sfc_temperature_anomaly.pdf",
)
# %%

projection = ccrs.Robinson(central_longitude=10)
fig, ax = plt.subplots(
    figsize=(8, 4), subplot_kw={"projection": projection}, constrained_layout=True
)
ax.set_extent([-67, -10, -2, 22], crs=ccrs.PlateCarree())

egh.healpix_show(jra2tocean.mean("year"), ax=ax, cmap=cmocean.cm.thermal)
ax.add_feature(cf.COASTLINE, linewidth=0.8)
ax.add_feature(cf.BORDERS, linewidth=0.4)
