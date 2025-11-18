#%%+
import xarray as xr
import glob
import cmocean as cmo
import numpy as np
import seaborn as sns
from xhistogram.xarray import histogram
import easygems.healpix as egh
import matplotlib.pyplot as plt
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp
import myutils.data_helper as dh
import myutils.open_datasets as od


#%%
sea_mask = xr.open_dataset("/work/mh0492/m301067/orcestra/healpix/bc-land/orcestra_1250m_bc_land_hpz12.zarr", engine="zarr")
lam_2d = xr.open_mfdataset(glob.glob("/work/mh0492/m301067/orcestra/healpix/**-rerun/orcestra_1250m_*-rerun_2d_hpz12.zarr"), engine="zarr", parallel=True)

#%%
ocean = lam_2d.where(sea_mask.sea == 1)
ocean = egh.attach_coords(ocean)


#%%
cids = od.get_cids()
beach = od.open_dropsondes(cids["dropsondes"]).pipe(dh.interpolate_gaps).pipe(dh.extrapolate_sfc).sel(sonde=slice(2, None))
#%%
east = ocean.where(egh.get_extent_mask(
    ocean,
    extent=(-34, -20, 3.5, 13.5), # w, e, s, n
))
west = ocean.where(egh.get_extent_mask(
    ocean,
    extent=(-59, -45, 6, 16)
))

#%%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
fig, ax= plt.subplots(
    subplot_kw={"projection": ccrs.EqualEarth(-135.58)},
)
ax.set_extent((-62, -15, 0, 20))

ax.coastlines()
p = egh.healpix_show(
        east.ts.isel(time=0),  ax=ax
)
egh.healpix_show(
        west.ts.isel(time=0),  ax=ax
)
#%%
from pydropsonde.helper import physics
from xhistogram.xarray import histogram
#%%

precip_bin =  np.insert(np.geomspace(1e-5, 20, 50), 0, 0)
hist_total = histogram(
    ocean.prw, ocean.pr * 3600,
    bins = [np.linspace(30,90, 30), precip_bin]
).compute()
#%%
#%%
hist_east = histogram(
    east.prw, east.pr * 3600,
    bins = [np.linspace(30,90, 30), precip_bin]
).compute()
hist_west = histogram(
    west.prw, west.pr * 3600,
    bins = [np.linspace(30,90, 30), precip_bin]
).compute()
#%%

imerg = xr.open_dataset(
    "/scratch/m/m301046/imerg_sondes.zarr"
)
#%%
ds = beach.set_coords("sonde_id").swap_dims({"sonde":"sonde_id"}).assign(
    pr = imerg.reset_coords(["lat", "lon", "time"]).calibrated_precipitation
)
hist_ds = ds.swap_dims({"sonde_id":"iwv"})
beach_hist = histogram(
    hist_ds.iwv,
    hist_ds.pr,
    bins = [np.linspace(40,75, 30), precip_bin]
)

#%%

#((plt_hist * plt_hist["rh_bin"]) / plt_hist.sum("rh_bin")).sum("rh_bin")
fig, ax = plt.subplots()
((hist_total * hist_total.pr_bin)/ hist_total.sum("pr_bin")).sum("pr_bin").plot(ax=ax, color="k", label="LAM total")
((hist_east * hist_east.pr_bin)/ hist_east.sum("pr_bin")).sum("pr_bin").plot(ax=ax, color="orange", label="LAM east")
((hist_west * hist_west.pr_bin)/ hist_west.sum("pr_bin")).sum("pr_bin").plot(ax=ax, color="green", label="LAM west")
ax.scatter(
    beach.sortby("sonde_id").iwv,
    imerg.calibrated_precipitation.sortby("sonde_id"),
    color="royalblue",
    alpha=0.2,
    s=7,
    label="BEACH/IMERG",
)
((beach_hist * beach_hist.pr_bin)/ beach_hist.sum("pr_bin")).sum("pr_bin").isel(iwv_bin=slice(7, -5)).plot(ax=ax, color="royalblue", label="BEACH")

sns.despine()
ax.set_yscale("log")
ax.set_xlim(40, 70)
ax.set_ylim(0, 20)
ax.set_xlabel("IWV / kg m-2")
ax.set_ylabel("pr / mm h-1")
ax.legend()
