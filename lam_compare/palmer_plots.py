# %%
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp
import moist_thermodynamics.constants as mtc
import sys

sys.path.append("/home/m/m301046/code/mid_level_clouds/mlclouds/")
import myutils.data_helper as dh
import myutils.open_datasets as od

# %%
mse_sat_hist = xr.open_dataset(
    "/scratch/m/m301046/lam_compare/lam_mse_sat_hist.zarr", engine="zarr"
)

mse_hist_full = xr.open_dataset(
    "/scratch/m/m301046/lam_compare/lam_mse_sat_hist_all.zarr", engine="zarr"
)
mse_sat_hist = mse_sat_hist.assign_coords(
    delta_mse=-mse_sat_hist.delta_mse_bin
).swap_dims({"delta_mse_bin": "delta_mse"})

mse_hist_full = mse_hist_full.assign_coords(
    delta_mse=-mse_hist_full.delta_mse_bin
).swap_dims({"delta_mse_bin": "delta_mse"})
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
lamsondes = xr.open_dataset(
    "/scratch/m/m301046/lam_sondes_z.zarr", engine="zarr"
).sortby("z")

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

lamsondes = lamsondes.assign(
    mse=mtf.moist_static_energy(
        T=lamsondes.ta,
        Z=lamsondes.z,
        qv=mtf.partial_pressure_to_specific_humidity(
            svp.liq_murphy_koop(lamsondes.ta), lamsondes.pfull
        ),
    ),
    sat_def=mtc.lv0
    * (
        mtf.partial_pressure_to_specific_humidity(
            svp.liq_murphy_koop(lamsondes.ta), lamsondes.pfull
        )
        - lamsondes.qv
    ),
)

lamsondes = lamsondes.assign(
    delta_mse=(
        lamsondes.mse.sel(z=heights[0], method="nearest")
        - lamsondes.mse.sel(z=heights[1], method="nearest")
    )
    * 1e-3,
    sat_def_mean=lamsondes.sat_def.sel(z=slice(heights[0], heights[1])).mean("z")
    * 1e-3,
)

# %%
# %%
dz = heights[1] - heights[0]
eps = 0.5e-3
heps = np.arange(-20, -3, 0.1)
qeps = -heps / dz / eps

bins_mse = np.linspace(-30, 10, 50)
bins_q = np.linspace(-10, 25, 50)

fig, ax = plt.subplots(figsize=(6, 6))


mse_hist_full.histogram_delta_mse_sat_def.plot.contourf(
    cmap="Reds", ax=ax, levels=10, add_colorbar=False
)

"""
sns.kdeplot(
    ax=ax, x=orc.sat_def_mean, y=-orc.delta_mse, levels=10, cmap="Blues", cbar=False
)

sns.kdeplot(
    ax=ax, x=lamsondes.sat_def_mean, y=-lamsondes.delta_mse, levels=10, cmap="Reds", cbar=False
)

ax.scatter(
    beach.sat_def_mean,
    -beach.delta_mse,
    color="black",
    alpha=0.2,
    s=2,
)
"""
ax.plot(qeps, heps, label=f"$\\varepsilon = {eps * 1000}$ km $^{{-1}}$", color="orange")
ax.set_xlim(0, 15)
ax.set_ylim(-25, 0)
ax.legend()
ax.set_xlabel("saturation deficit / kJ kg$^{-1}$")
ax.set_ylabel("$\Delta$ MSE$^*$ / kJ kg$^{-1}$")
sns.despine(offset={"bottom": 5})
fig.savefig("/scratch/m/m301046/lam_palmer_singh.pdf")
