#%%
import xarray as xr
import glob
import cmocean as cmo
import numpy as np
import seaborn as sns
from xhistogram.xarray import histogram
import matplotlib.pyplot as plt
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp
import moist_thermodynamics.constants as mtc
import myutils.data_helper as dh
import myutils.open_datasets as od

#%%

path_to_files = glob.glob("/work/mh0492/m301067/orcestra/healpix/**-rerun/orcestra_1250m_*-rerun_3d_hpz12.zarr")
lam_3d=xr.open_mfdataset(path_to_files, engine="zarr", parallel=True, chunks={"time":12, "height_full":-1, "cell":16384})

#%%
grid = xr.open_dataset(
    "/work/mh0492/m301067/orcestra/auxiliary-files/grids/ORCESTRA_1250m_DOM01_vgrid.nc"
)
heights = grid.where((np.rad2deg(grid.clon) > -40)).where((np.rad2deg(grid.clon) < -30)).where((np.rad2deg(grid.clat) < 10)).where((np.rad2deg(grid.clat) >8)).mean("ncells")
heights.to_zarr("/scratch/m/m301046/lam_heights.zarr", mode="w", zarr_format=2)
#%%
heights = xr.open_dataset("/scratch/m/m301046/lam_heights.zarr", engine="zarr")
#%%
sea_mask = xr.open_dataset("/work/mh0492/m301067/orcestra/healpix/bc-land/orcestra_1250m_bc_land_hpz12.zarr", engine="zarr")
lam_3d = lam_3d.where(sea_mask.sea == 1)
#%%
lam_3d = lam_3d.assign(
    z = heights.zg.rename({"height_2":"height_full"}),
    z_half = heights.zghalf.rename({"height":"height_half"}),
).swap_dims({"height_full":"z", "height_half":"z_half"})
#%% pfull mean 850 : z ~1500m -12 - mean 500 m : z~6000m -26

lam_3d = lam_3d.assign(
    mse = mtf.moist_static_energy(
        T=lam_3d.ta,
        Z=lam_3d.z,
        qv=lam_3d.qv,
    ),
    sat_def = mtc.lv0*(mtf.relative_humidity_to_specific_humidity(
        RH=1, 
        p=lam_3d.pfull,
        T=lam_3d.ta,
        es = svp.liq_wagner_pruss
    ) - lam_3d.qv)
)
#%%

delta_mse = (lam_3d.mse.sel(z=1500, method="nearest") - lam_3d.mse.sel(z=6000, method="nearest")).rename("delta_mse") * 1e-3
delta_q = lam_3d.sat_def.isel(z=slice(-26, -11)).mean("z") * 1e-3

#%%

bins_mse = np.linspace(-10, 30, 50)
bins_q = np.linspace(-10, 25, 50)

mse_sat_hist = histogram(
    delta_mse, delta_q,
    bins=[bins_mse, bins_q]
)
mse_sat_hist.to_zarr("/scratch/m/m301046/lam_compare/lam_mse_sat_hist.zarr", mode="w", zarr_format=2)

#%%

mse_sat_hist = xr.open_dataset("/scratch/m/m301046/lam_compare/lam_mse_sat_hist.zarr", engine="zarr")

#%%
#%%
beach = od.open_dropsondes(od.get_cids()["dropsondes"]).pipe(dh.interpolate_gaps).pipe(dh.extrapolate_sfc).sel(sonde=slice(2, None))

beach = beach.assign(
    mse = mtf.moist_static_energy(
        T=beach.ta,
        Z=beach.altitude,
        qv=beach.q,
    ),
    sat_def = mtc.lv0*(mtf.relative_humidity_to_specific_humidity(
        RH=1, 
        p=beach.p,
        T=beach.ta,
        es = svp.liq_hardy
    ) - beach.q)
)

beach = beach.assign( 
    delta_mse = (beach.mse.sel(altitude=1500, method="nearest") - beach.mse.sel(altitude=6000, method="nearest")) * 1e-3,
    sat_def_mean = beach.sat_def.sel(altitude=slice(1500, 6000)).mean("altitude") * 1e-3
)
#%%
mse_sat_hist = mse_sat_hist.assign_coords(delta_mse=-mse_sat_hist.delta_mse_bin).swap_dims({
    "delta_mse_bin": "delta_mse"
})
#%%

bins_mse = np.linspace(-30, 10, 50)
bins_q = np.linspace(-10, 25, 50)


mse_sat_hist.histogram_delta_mse_sat_def.plot(cmap="Blues")

plt.scatter(
     beach.sat_def_mean,
    -beach.delta_mse,
    color="black",
    alpha=0.2,
    s=2,
)


plt.xlim(0, 25)
plt.ylim(-30, 10)
plt.xlabel("saturation deficit / kJ kg$^{-1}$")
plt.ylabel("$\Delta$ MSE / kJ kg$^{-1}$")