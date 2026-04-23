#! /usr/bin/env python3
#SBATCH --account=mh0066
#SBATCH --partition=compute
#SBATCH --time=08:00:00

#%%
import xarray as xr
import intake
import easygems.healpix as egh
import moist_thermodynamics.functions as mtf
import matplotlib.pyplot as plt
import seaborn as sns
#%%
cat = intake.open_catalog("https://tcodata.mpimet.mpg.de/internal.yaml")
ds = cat.ORCESTRA.LAM_ORCESTRA(dim="3d").to_dask().pipe(egh.attach_coords)

ds = ds.assign(
    theta_e_all = mtf.theta_e(ds.ta, ds.pfull, ds.qc + ds.qv + ds.qg + ds.qr + ds.qi + ds.qs),
)
# %%

# %%

#%%
east = [[-34.5, 13.5], [-20.0, 13.5], [-20.0, 2.5], [-34.5, 2.5]]
west = [[-59, 17], [-44.5, 17], [-44.5, 6],[-59, 6] ]

east_coords = [east[0][0] % 360, east[1][0] %360, east[2][1], east[0][1]]
west_coords = [west[0][0] % 360, west[1][0] %360,west[2][1], west[0][1]]

east_mask = egh.get_extent_mask(
    ds, east_coords
)
west_mask = egh.get_extent_mask(
    ds, west_coords
)
ds_east = ds.sel(cell=east_mask)
ds_west = ds.sel(cell=west_mask)

#%%
ds_east.chunk({"time":24, "height_full":4, "cell":16384})[["ta", "theta_e_all",  "qv"]].to_zarr("/scratch/m/m301046/east_theta_e_lam.zarr", mode="w")
#%%
ds_west.chunk({"time":24, "height_full":4, "cell":16384})[["ta", "theta_e_all",  "qv"]].to_zarr("/scratch/m/m301046/west_theta_e_lam.zarr", mode="w")
