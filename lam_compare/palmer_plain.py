#!/usr/bin/env python3
# SBATCH --account=mh0066
# SBATCH --partition=compute
# SBATCH --time=04:00:00
# %%
import intake
import xarray as xr
import numpy as np
import seaborn as sns
from xhistogram.xarray import histogram
import matplotlib.pyplot as plt
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp
import moist_thermodynamics.constants as mtc
import sys

sys.path.append("/home/m/m301046/code/mid_level_clouds/mlclouds/")
import myutils.data_helper as dh
import myutils.open_datasets as od
import easygems.healpix as egh
import cartopy.crs as ccrs
import cartopy.feature as cf

# %%
cat = intake.open_catalog("https://tcodata.mpimet.mpg.de/internal.yaml")
lam_2d = cat.ORCESTRA.LAM_ORCESTRA.to_dask()
lam_3d = cat.ORCESTRA.LAM_ORCESTRA(dim="3d").to_dask()


# %%
lam_3d = lam_3d.where(lam_3d.sea == 1)
# %%
lam_3d = lam_3d.assign(
    z=lam_3d.zg.mean("cell").rename("z"),
    z_half=lam_3d.zghalf.mean("cell").rename("z_half"),
).swap_dims({"height_full": "z", "height_half": "z_half"})
# %% pfull mean 850 : z ~1500m -12 - mean 500 m : z~6000m -26

zoom6 = lam_3d.coarsen(cell=4**6).mean().resample(time="1D").mean()
zoom6["crs"].attrs = dict(
    grid_mapping_name="healpix",
    healpix_nside=int(zoom6["crs"].attrs["healpix_nside"] // 2**6),
    healpix_order="nest",
)
zoom6 = zoom6.assign_coords(cell=zoom6.cell // 4**6)
# %%
lam2d6 = lam_2d.coarsen(cell=4**6).mean().resample(time="1D").mean()
lam2d6["crs"].attrs = dict(
    grid_mapping_name="healpix",
    healpix_nside=int(lam2d6["crs"].attrs["healpix_nside"] // 2**6),
    healpix_order="nest",
)

lam2d6 = lam2d6.assign_coords(cell=lam2d6.cell // 4**6)
# %%

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
# %%

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
# %%

delta_mse = (
    ds.mse.sel(z=1500, method="nearest") - ds.mse.sel(z=6000, method="nearest")
).rename("delta_mse") * 1e-3
delta_q = ds.sat_def.isel(z=slice(-26, -11)).mean("z") * 1e-3
# %%


# %%

bins_mse = np.linspace(-10, 30, 50)
bins_q = np.linspace(-10, 25, 50)

mse_sat_hist = histogram(
    delta_mse.where(lam2d6.pr * 86400 > 5),
    delta_q.where(lam2d6.pr * 86400 > 5),
    bins=[bins_mse, bins_q],
)
mse_sat_hist.to_zarr(
    "/scratch/m/m301046/lam_compare/lam_mse_sat_hist.zarr", mode="w", zarr_format=2
)

# %%
mse_sat_hist_all = histogram(delta_mse, delta_q, bins=[bins_mse, bins_q])
mse_sat_hist_all.to_zarr(
    "/scratch/m/m301046/lam_compare/lam_mse_sat_hist_all.zarr", mode="w", zarr_format=2
)

# %%

mse_sat_hist = xr.open_dataset(
    "/scratch/m/m301046/lam_compare/lam_mse_sat_hist.zarr", engine="zarr"
)

# %%
# %%
beach = (
    od.open_dropsondes(od.get_cids()["dropsondes"])
    .pipe(dh.interpolate_gaps)
    .pipe(dh.extrapolate_sfc)
    .sel(sonde=slice(2, None))
    .reset_coords(["launch_lat", "launch_lon"])
)
rapsodi = (
    od.open_radiosondes(od.get_cids()["radiosondes"])
    .pipe(dh.interpolate_gaps)
    .pipe(dh.extrapolate_sfc)
)
orcestra = xr.concat([beach, rapsodi], dim="sonde")

# %%
heights = [1518, 5783]  # (1500, 6000)

orcestra = dh.sel_sub_domain(orcestra, dh.east)
orc = orcestra.assign(
    mse=mtf.moist_static_energy(
        T=orcestra.ta,
        Z=orcestra.altitude,
        qv=mtf.partial_pressure_to_specific_humidity(
            svp.liq_murphy_koop(orcestra.ta), orcestra.p
        ),
    ),
    sat_def=mtc.lv0
    * (
        mtf.partial_pressure_to_specific_humidity(
            svp.liq_murphy_koop(orcestra.ta), orcestra.p
        )
        - orcestra.q
    ),
)

orc = orc.assign(
    delta_mse=(
        orc.mse.sel(altitude=heights[0], method="nearest")
        - orc.mse.sel(altitude=heights[1], method="nearest")
    )
    * 1e-3,
    sat_def_mean=orc.sat_def.sel(altitude=slice(heights[0], heights[1])).mean(
        "altitude"
    )
    * 1e-3,
)
# %%
mse_sat_hist = mse_sat_hist.assign_coords(
    delta_mse=-mse_sat_hist.delta_mse_bin
).swap_dims({"delta_mse_bin": "delta_mse"})
# %%
dz = heights[1] - heights[0]
eps = 0.5e-3
heps = np.arange(-20, -3, 0.1)
qeps = -heps / dz / eps

bins_mse = np.linspace(-30, 10, 50)
bins_q = np.linspace(-10, 25, 50)

fig, ax = plt.subplots(figsize=(6, 6))
mse_sat_hist.histogram_delta_mse_sat_def.plot.contourf(
    cmap="Blues", ax=ax, levels=10, add_colorbar=False
)

sns.kdeplot(
    ax=ax, x=orc.sat_def_mean, y=-orc.delta_mse, levels=15, cmap="Blues", cbar=False
)

"""
ax.scatter(
    beach.sat_def_mean,
    -beach.delta_mse,
    color="black",
    alpha=0.2,
    s=2,
)
"""
ax.plot(qeps, heps, label=f"$\\varepsilon = {eps * 1000}$ km $^{{-1}}$")
ax.set_xlim(0, 20)
ax.set_ylim(-25, 5)
ax.set_xlabel("saturation deficit / kJ kg$^{-1}$")
ax.set_ylabel("$\Delta$ MSE / kJ kg$^{-1}$")
sns.despine(offset={"bottom": 5})
fig.savefig("/scratch/m/m301046/lam_palmer_singh.pdf")
