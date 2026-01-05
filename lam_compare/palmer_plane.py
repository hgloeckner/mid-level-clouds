#!/usr/bin/env python3
# SBATCH --account=mh0066
# SBATCH --partition=compute
# SBATCH --time=04:00:00
# %%
import xarray as xr
import numpy as np
from xhistogram.xarray import histogram
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp
import moist_thermodynamics.constants as mtc
import sys

sys.path.append("/home/m/m301046/code/mid_level_clouds/mlclouds/")

# %%
"""
cat = intake.open_catalog("https://tcodata.mpimet.mpg.de/internal.yaml")
#lam_2d = cat.ORCESTRA.LAM_ORCESTRA.to_dask()
lam_3d = cat.ORCESTRA.LAM_ORCESTRA(dim="3d").to_dask()

lam_3d = lam_3d.where(lam_3d.sea == 1)
"""
# %%
"""
lam_3d = lam_3d.assign(
    z=lam_3d.zg.mean("cell").rename("z"),
    z_half=lam_3d.zghalf.mean("cell").rename("z_half"),
).swap_dims({"height_full": "z", "height_half": "z_half"})
"""
# %% pfull mean 850 : z ~1500m -12 - mean 500 m : z~6000m -26
"""
zoom6 = lam_3d.coarsen(cell=4**6).mean().resample(time="1D").mean()
zoom6["crs"].attrs = dict(
    grid_mapping_name="healpix",
    healpix_nside=int(zoom6["crs"].attrs["healpix_nside"] // 2**6),
    healpix_order="nest",
)
zoom6 = zoom6.assign_coords(cell=zoom6.cell // 4**6)
"""
# %%
"""
lam2d6 = lam_2d.coarsen(cell=4**6).mean().resample(time="1D").mean()
lam2d6["crs"].attrs = dict(
    grid_mapping_name="healpix",
    healpix_nside=int(lam2d6["crs"].attrs["healpix_nside"] // 2**6),
    healpix_order="nest",
)

lam2d6 = lam2d6.assign_coords(cell=lam2d6.cell // 4**6)
"""
# %%
"""
projection = ccrs.Robinson(central_longitude=10)
fig, axes = plt.subplots(
    figsize=(10, 10), nrows=2, subplot_kw={"projection": projection}
)
for ax in axes:
    ax.set_extent([-62, -15, 2, 20], crs=ccrs.PlateCarree())

egh.healpix_show(lam_3d.resample(time="1D").mean().isel(time=10, z=-1).ta, ax=axes[0])
egh.healpix_show(zoom6.isel(time=10, z=-1).ta, ax=axes[1])
for ax in axes:
    ax.add_feature(cf.COASTLINE, lw=0.5)
"""
# %%
"""
qs = mtf.partial_pressure_to_specific_humidity(
    svp.liq_murphy_koop(zoom6.ta), zoom6.pfull
)
ds = zoom6.assign(
    mse=mtf.moist_static_energy(
        T=zoom6.ta,
        Z=zoom6.z,
        qv=qs,
    ),
    sat_def=mtc.lv0 * (qs - zoom6.qv),
)
"""

# %%

ds = xr.open_dataset("/scratch/m/m301046/lam_sondes_z.zarr", engine="zarr")
ds = ds.assign(
    mse=mtf.moist_static_energy(
        T=ds.ta,
        Z=ds.z,
        qv=mtf.partial_pressure_to_specific_humidity(
            svp.liq_murphy_koop(ds.ta), ds.pfull
        ),
    ),
    sat_def=mtc.lv0
    * (
        mtf.partial_pressure_to_specific_humidity(svp.liq_murphy_koop(ds.ta), ds.pfull)
        - ds.qv
    ),
)
# %%

delta_mse = (
    ds.mse.sel(z=1500, method="nearest") - ds.mse.sel(z=6000, method="nearest")
).rename("delta_mse") * 1e-3
delta_q = ds.sat_def.isel(z=slice(-26, -11)).mean("z") * 1e-3

# %%


# %%

bins_mse = np.linspace(-10, 30, 50)
bins_q = np.linspace(-10, 25, 50)
"""
mse_sat_hist = histogram(
    delta_mse.where(lam2d6.pr * 86400 > 5),
    delta_q.where(lam2d6.pr * 86400 > 5),
    bins=[bins_mse, bins_q],
)
mse_sat_hist.to_zarr(
    "/scratch/m/m301046/lam_compare/lam_mse_sat_hist.zarr", mode="w", zarr_format=2
)

mse_sat_hist_all = histogram(delta_mse, delta_q, bins=[bins_mse, bins_q])
mse_sat_hist_all.to_zarr(
    "/scratch/m/m301046/lam_compare/lam_mse_sat_hist_all.zarr", mode="w", zarr_format=2
)
"""
# %%
mse_sat_hist_sondes = histogram(delta_mse, delta_q, bins=[bins_mse, bins_q])
mse_sat_hist_sondes.to_zarr(
    "/scratch/m/m301046/lam_compare/lam_mse_sat_hist_sondes.zarr",
    mode="w",
    zarr_format=2,
)
