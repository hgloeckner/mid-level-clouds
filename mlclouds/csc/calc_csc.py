
#%%
import numpy as np
import xarray as xr

import myutils.physics_helper as ph

rrtmg = xr.open_dataset(
    "/work/mh0066/m301046/data/mlclouds/idealized_rrtmg_fluxes.zarr"
)
idealized = xr.open_dataset(
    "/work/mh0066/m301046/data/mlclouds/idealized_profiles.nc"
)
rrtmg = rrtmg.assign(
    lw_htgr=(("sonde", "level"), xr.apply_ufunc(
        ph.calc_heating_rate_from_flx,
        rrtmg.lw_flux_up,
        rrtmg.lw_flux_down,
        rrtmg.pres_level,
        input_core_dims=[["level"], ["level"], ["level"]],
        output_core_dims=[["level"]],
        vectorize=True,
        #dask="parallelized",
    ).values),
    sw_htgr=(("sonde", "level"), xr.apply_ufunc(
        ph.calc_heating_rate_from_flx,      
        rrtmg.sw_flux_up,
        rrtmg.sw_flux_down,
        rrtmg.pres_level,
        input_core_dims=[["level"], ["level"], ["level"]],
        output_core_dims=[["level"]],
        vectorize=True,
        #dask="parallelized",
    ).values),
    sonde = ("sonde", idealized.sonde.values),
    theta = (("sonde", "level"), idealized.theta.values),
    q = (("sonde", "level"), idealized.q.values),
).swap_dims({"level": "altitude"}).rename({
    "pres_level":"p",
    "temp_level":"ta",
})

#%%
arts = xr.open_dataset(
    "/work/mh0066/m301046/data/mlclouds/idealized_arts.zarr"
)
arts = xr.merge(
    [
        arts,
        idealized,
    ],
    compat="override",
).rename(
    {"lw_heating_rate": "lw_htgr", "sw_heating_rate": "sw_htgr"}
)
arts = arts.assign(
    lw_htgr=arts.lw_htgr / 60 / 60 / 24,
)


def calc_cs_convergence(ds, ta_var="ta", qvar="q"):
    res = {}
    res["stab"] = ph.get_stability(ds.theta, ds[ta_var])  # .rolling(altitude=10).mean()
    res["rho"] = ph.density_from_q(
        ds.p, ds[ta_var], ds[qvar]
    )  # .rolling(altitude=10).mean()
    htgr = ds.lw_htgr  # .rolling(altitude=10).mean()
    res["csc_stab_lw"] = ph.get_csc_stab(res["rho"], res["stab"], htgr)
    res["csc_cool_lw"] = ph.get_csc_cooling(res["rho"], res["stab"], htgr)
    res["lw_mass_flux"] = ph.mass_flux(res["rho"], res["stab"], htgr)
    return res
    htgr = ds.lw_htgr + ds.sw_htgr.mean("mu0")  # .rolling(altitude=10).mean()
    res["csc_stab"] = ph.get_csc_stab(res["rho"], res["stab"], htgr)
    res["csc_cool"] = ph.get_csc_cooling(res["rho"], res["stab"], htgr)
    res["mass_flux"] = ph.mass_flux(res["rho"], res["stab"], htgr)
    htgr = ds.lw_htgr + ds.sw_htgr.sel(mu0=12)  # .rolling(altitude=10).mean()
    res["csc_stab_swmax"] = ph.get_csc_stab(res["rho"], res["stab"], htgr)
    res["csc_cool_swmax"] = ph.get_csc_cooling(res["rho"], res["stab"], htgr)
    return res


def get_heating_from_mass_flux(mass_flux, stability, rho):
    return mass_flux * stability / rho
#%%
resrrtmg = {}
resarts = {}
for shape in rrtmg.sonde.values:
    resrrtmg[shape] = calc_cs_convergence(rrtmg.sel(sonde=shape), ta_var="ta", qvar="q")
for shape in arts.sonde.values:
    resarts[shape] = calc_cs_convergence(arts.sel(sonde=shape), ta_var="ta", qvar="q")

#%%
import matplotlib.pyplot as plt
import seaborn as sns
color = "#006C66"
altslice=slice(0, 12000)
sns.set_palette("Paired")
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 8), sharey=True)

for shape, ls in [("clow", ":"), ("elow", "-")]:
    for ds, resds, name in [(rrtmg, resrrtmg, "rrtmg"), (arts, resarts, "arts")]:
        for i, var in enumerate([ds.lw_htgr.sel(sonde=shape), resds[shape]["csc_cool_lw"], resds[shape]["csc_cool_lw"] + resds[shape]["csc_stab_lw"]]):
            ax[i].plot(
                var.sel(altitude=altslice) * 60 * 60 * 24,
                var.altitude.sel(altitude=altslice),
                
        label=name + " " + shape,
        ls=ls,
    )

ax[1].set_xlim(-0.3, 0.3)
ax[2].set_xlim(-0.3, 0.3)
ax[0].legend()
for ax in ax:
    ax.axvline(0, color="k", ls="-", linewidth=0.5, alpha=0.5)
sns.despine()