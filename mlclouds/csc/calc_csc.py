# %%
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import myutils.physics_helper as ph

idealized = xr.open_dataset("/work/mh0066/m301046/data/mlclouds/idealized_profiles.nc")
arts = (
    xr.open_dataset(
        "/work/mh0066/m301046/data/idealized_arts_fluxes.zarr", engine="zarr", chunks={}
    )
    .load()
    .rename(
        {
            "sw_heating_rate": "htgr_sw",
            "lw_heating_rate": "htgr_lw",
        }
    )
)
arts = arts.assign(
    htgr_sw=arts.htgr_sw.fillna(0) / (60 * 60 * 24),
    htgr_lw=arts.htgr_lw.fillna(0) / (60 * 60 * 24),
).assign(htgr_sw_mean=arts.htgr_sw.fillna(0).mean("hour_of_day") / (60 * 60 * 24))
arts = xr.merge([arts, idealized], compat="override")
rrtmg = xr.open_dataset(
    "/work/mh0066/m301046/data/idealized_rrtmg_fluxes.zarr", engine="zarr", chunks={}
).load()
rrtmg = xr.merge([rrtmg, idealized], compat="override")
rrtmg = rrtmg.assign(
    sw_flux_up_mean=rrtmg.sw_flux_up.mean("mu0"),
    sw_flux_down_mean=rrtmg.sw_flux_down.mean("mu0"),
)
rrtmg = rrtmg.assign(
    htgr_lw=xr.apply_ufunc(
        ph.calc_heating_rate_from_flx,
        rrtmg.lw_flux_up,
        rrtmg.lw_flux_down,
        rrtmg.pres_level,
        input_core_dims=[["altitude"], ["altitude"], ["altitude"]],
        output_core_dims=[["altitude"]],
        vectorize=True,
    ),
    htgr_sw_mean=xr.apply_ufunc(
        ph.calc_heating_rate_from_flx,
        rrtmg.sw_flux_up_mean,
        rrtmg.sw_flux_down_mean,
        rrtmg.pres_level,
        input_core_dims=[["altitude"], ["altitude"], ["altitude"]],
        output_core_dims=[["altitude"]],
        vectorize=True,
    ),
    htgr_sw=xr.apply_ufunc(
        ph.calc_heating_rate_from_flx,
        rrtmg.sw_flux_up,
        rrtmg.sw_flux_down,
        rrtmg.pres_level,
        input_core_dims=[["altitude"], ["altitude"], ["altitude"]],
        output_core_dims=[["altitude"]],
        vectorize=True,
    ),
)
rrtmg = rrtmg.assign(
    htgr=rrtmg.htgr_lw + rrtmg.htgr_sw.mean("mu0"),
    htgr_sw_high=rrtmg.htgr_sw.isel(mu0=14),
)
arts = arts.assign(
    htgr=arts.htgr_lw + arts.htgr_sw.mean("hour_of_day"),
    htgr_sw_high=arts.htgr_sw.sel(hour_of_day=12),
)


# %%
def calc_cs_convergence(ds, ta_var="ta", qvar="q"):
    ds = ds.assign(
        stab=ph.get_stability(ds.theta, ds[ta_var]),
        rho=ph.density_from_q(ds.p, ds[ta_var], ds[qvar]),
    )
    for htgr in [(ds.htgr, ""), (ds.htgr_lw, "_lw"), (ds.htgr_sw_mean, "_sw_mean")]:
        ds = ds.assign(
            {
                f"csc_stab{htgr[1]}": ph.get_csc_stab(ds.rho, ds.stab, htgr[0]),
                f"csc_cool{htgr[1]}": ph.get_csc_cooling(ds.rho, ds.stab, htgr[0]),
                f"mass_flux{htgr[1]}": ph.mass_flux(ds.rho, ds.stab, htgr[0]),
            }
        )
    return ds


cscarts = calc_cs_convergence(arts)
cscrrtmg = calc_cs_convergence(rrtmg)
# %%

htgr = ""
sondes = ["cmid", "emid"]

fig, axes = plt.subplots(ncols=3, figsize=(15, 6))
for sonde in sondes:
    pltarts = cscarts.sel(sonde=sonde).sel(altitude=slice(0, 12000))
    pltrrtmg = cscrrtmg.sel(sonde=sonde).sel(altitude=slice(0, 12000))
    for ds, ls in [(pltarts, "-"), (pltrrtmg, "--")]:
        axes[0].plot(ds[f"htgr{htgr}"] * 60 * 60 * 24, ds.T, ls=ls)
        axes[1].plot(ds[f"csc_cool{htgr}"] * 60 * 60 * 24, ds.T, ls=ls)
        axes[2].plot(
            (ds[f"csc_cool{htgr}"] + ds[f"csc_stab{htgr}"]) * 60 * 60 * 24, ds.T, ls=ls
        )
for ax in axes:
    ax.invert_yaxis()
    ax.set_ylim(290, 250)
    ax.axvline(0, color="k", ls=":", linewidth=1)
axes[1].set_xlim(-0.5, 0.5)
axes[2].set_xlim(-0.5, 0.5)
sns.despine()

# %%
cw = 8.27
sns.set_context("paper")
fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(cw, cw * 0.5))
