# %%

import sys

sys.path.append("../../")
import myutils.physics_helper as ph
import angles
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rad_helper as rad
import myutils.data_helper as dh
from pyrte_rrtmgp.rrtmgp import GasOptics
from pyrte_rrtmgp.rrtmgp_data_files import GasOpticsFiles

# %%
ds = xr.open_dataset(
    "/Users/helene/Documents/Data/mlclouds/sondes_for_radiation.nc"
).swap_dims({"sonde": "sonde_id"})
lvl3 = xr.open_dataset(
    "ipfs://bafybeiesyutuduzqwvu4ydn7ktihjljicywxeth6wtgd5zi4ynxzqngx4m", engine="zarr"
).swap_dims({"sonde": "sonde_id"})
ds = ds.assign(
    launch_lat=lvl3.launch_lat.sel(sonde_id=ds.sonde_id),
    launch_lon=lvl3.launch_lon.sel(sonde_id=ds.sonde_id),
    launch_time=lvl3.launch_time.sel(sonde_id=ds.sonde_id),
).swap_dims({"sonde_id": "sonde"})
# %%

# %%
ds = ds.assign(
    mu0=(
        "sonde",
        xr.apply_ufunc(
            angles.cos_zenith_angle,
            ds.launch_time,
            ds.launch_lat,
            ds.launch_lon,
            vectorize=True,
        ).values,
        {"long_name": "cosine of solar zenith angle at launch time", "units": "1"},
    ),
)

# %%
es = mtf.make_es_mxd(svp.liq_wagner_pruss, svp.ice_wagner_etal)
mu0 = angles.get_mu_day(np.datetime64("2024-08-30T00:00:00"), lat=0, lon=-30)

# %%

plt.plot((angles.get_hour_from_time(ds.launch_time)), ds.mu0, linestyle="", marker="o")

# %%
Psfc = ds.p.mean(dim="sonde").sel(altitude=0).values
P = np.arange(Psfc, 4000.0, -500)
sfcT = ds.t.mean(dim="sonde").sel(altitude=0).values
qsfc = ds.q.mean(dim="sonde").sel(altitude=0).values  # 9182267570514704

pseudo = ph.make_sounding_from_adiabat(P, sfcT, qsfc, thx=mtf.theta_e_bolton, Tmin=195)

pseudo = xr.concat(
    [
        pseudo,
        ds[["t", "p"]]
        .mean("sonde")
        .rename({"t": "T", "p": "P"})
        .sel(altitude=slice(pseudo.altitude.max().values, None)),
    ],
    dim="altitude",
    compat="no_conflicts",
)
pseudo = pseudo.where(pseudo.T > 195).interpolate_na("altitude", method="akima")
plcl = mtf.plcl_bolton(T=sfcT, P=Psfc, qt=qsfc)
zlcl = mtf.zlcl(plcl, T=sfcT, P=Psfc, qt=qsfc, z=0)
# %%
rhs = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
quniform = {
    rh: rad.uniform_humidity(
        pseudo,
        zlcl=zlcl,
        ztoa=pseudo.altitude[pseudo.T.argmin()],
        rh=rh,
        es=es,
    )
    for rh in rhs
}

# %%


qcshape = {
    rh: rad.cshape_humidity(
        pseudo, zlcl=zlcl, rhmid=0.5 * rh, rhlcl=rh, rhtoa=rh, Tmin=265, es=es
    )
    for rh in rhs
}

qwshape = {
    rh: rad.wshape_humidity(
        pseudo, zlcl=zlcl, rhmid=0.5 * rh, rhlcl=rh, rhtoa=rh, Tmin=265, es=es
    )
    for rh in rhs
}


qs = qwshape
# %%
lvl3 = lvl3.pipe(dh.interpolate_gaps).pipe(dh.extrapolate_sfc)
fig, ax = plt.subplots()

ax.plot(
    mtf.specific_humidity_to_relative_humidity(qcshape[0.9], pseudo.P, pseudo.T, es=es),
    pseudo.T.values,
    label="cshape: RH 0.9",
)
ax.plot(
    mtf.specific_humidity_to_relative_humidity(qwshape[0.9], pseudo.P, pseudo.T, es=es),
    pseudo.T.values,
    label="wshape: RH 0.9",
)
ax.plot(
    mtf.specific_humidity_to_relative_humidity(
        lvl3.q.mean(dim="sonde_id").sel(altitude=slice(0, 12000)).values,
        lvl3.p.mean(dim="sonde_id").sel(altitude=slice(0, 12000)).values,
        lvl3.ta.mean(dim="sonde_id").sel(altitude=slice(0, 12000)).values,
        es=es,
    ),
    lvl3.ta.mean(dim="sonde_id").sel(altitude=slice(0, 12000)).values,
    color="k",
    label="mean dropsondes",
)
ax.legend()
ax.set_xlim(-0.1, 1)
# ax.set_yscale("log")
ax.invert_yaxis()
# %%
flxs = {"uniform": {}, "cshape": {}, "wshape": {}}
for qs, name in zip([quniform, qcshape, qwshape], ["uniform", "cshape", "wshape"]):
    for r, q in qs.items():
        atmosphere = rad.make_atmosphere(
            pseudo.P.values.reshape(1, pseudo.P.shape[0]),
            pseudo.T.values.reshape(1, pseudo.P.shape[0]),
            ph.specific_humidity2vmr(q).values.reshape(1, pseudo.P.shape[0]),
            o3=ds.O3.interp(altitude=pseudo.altitude).values,
        )
        gas_optics_lw = GasOptics(gas_optics_file=GasOpticsFiles.LW_G256)
        op_lw = gas_optics_lw.compute(atmosphere, add_to_input=False)
        op_lw = op_lw.assign(surface_emissivity=0.98)
        lw_fluxes = op_lw.rte.solve(add_to_input=False)

        gas_optics_sw = GasOptics(gas_optics_file=GasOpticsFiles.SW_G224)
        op_sw = gas_optics_sw.compute(atmosphere, add_to_input=False)
        op_sw["surface_albedo"] = 0.06
        sw_fluxes = []
        for mu in mu0:
            op_sw = op_sw.assign(mu0=mu)
            sw_fluxes.append(op_sw.rte.solve(add_to_input=False))

        flxs[name][r] = xr.merge(
            [lw_fluxes, xr.concat(sw_fluxes, dim="mu0").mean(dim="mu0"), atmosphere]
        )

# %%
lvl3rad = ds.where(
    ds.sonde_id.isin(ds.where(ds.mu0 > 0, drop=True).sonde_id),
    drop=True,
)
lvl3rad = lvl3rad.interp(altitude=pseudo.altitude)


atmosphere_sonde = rad.make_atmosphere(
    lvl3rad.p.mean(dim="sonde").values.reshape(1, lvl3rad.p.sizes["altitude"]),
    lvl3rad.t.mean(dim="sonde").values.reshape(1, lvl3rad.p.sizes["altitude"]),
    ph.specific_humidity2vmr(
        lvl3rad.q.mean(dim="sonde").values.reshape(1, lvl3rad.p.sizes["altitude"])
    ),
    o3=ds.O3.interp(altitude=pseudo.altitude).values,
)

atmosphere_sonde = atmosphere_sonde.assign(
    h2o=atmosphere_sonde.h2o.where(atmosphere_sonde.layer < 160).ffill(dim="layer")
)

gas_optics_lw = GasOptics(gas_optics_file=GasOpticsFiles.LW_G256)
oplw = gas_optics_lw.compute(atmosphere_sonde, add_to_input=False)
oplw = oplw.assign(surface_emissivity=0.98)

gas_optics_sw = GasOptics(gas_optics_file=GasOpticsFiles.SW_G224)
opsw = gas_optics_sw.compute(atmosphere_sonde, add_to_input=False)
flx = []
for mu in mu0:
    opsw = opsw.assign(surface_albedo=0.06, mu0=mu)
    flx.append(opsw.rte.solve(add_to_input=False))

flxs["sondes"] = {
    0.9: xr.merge(
        [
            oplw.rte.solve(add_to_input=False),
            xr.concat(flx, dim="mu0").mean(dim="mu0"),
            atmosphere_sonde,
        ]
    )
}
# %%
htgr = {"uniform": {}, "cshape": {}, "wshape": {}, "sondes": {}}
for name, atm in zip(
    flxs.keys(), [atmosphere, atmosphere, atmosphere, atmosphere_sonde]
):
    for r in flxs[name].keys():
        htgr[name][r] = {}
        htgr[name][r]["lw"] = (
            rad.calc_heating_rate_from_flx(
                flx_up=flxs[name][r].lw_flux_up.mean(dim="column"),
                flx_down=flxs[name][r].lw_flux_down.mean(dim="column"),
                p=atm.pres_level.mean(dim="column"),
            )
            * 60
            * 60
            * 24
        )
        htgr[name][r]["sw"] = (
            rad.calc_heating_rate_from_flx(
                flx_up=flxs[name][r].sw_flux_up.mean(dim="column"),
                flx_down=flxs[name][r].sw_flux_down.mean(dim="column"),
                p=atm.pres_level.mean(dim="column"),
            )
            * 60
            * 60
            * 24
        )
# %%

sw_colors = sns.color_palette("Reds", n_colors=len(rhs))
lw_colors = sns.color_palette("Blues", n_colors=len(rhs))
total_colors = sns.color_palette("Greys", n_colors=len(rhs))


fig, axs = plt.subplots(
    3, 2, width_ratios=(0.3, 0.7), figsize=(12, 12), sharex="col", sharey=True
)

for idx, name in enumerate(["uniform", "cshape", "wshape"]):
    axes = axs[idx]
    for i, r in enumerate(flxs[name].keys()):
        if r == 1.0:
            lw_label = "LW"
            sw_label = "SW"
        else:
            lw_label = None
            sw_label = None

        axes[1].plot(
            htgr[name][r]["lw"] + htgr[name][r]["sw"],
            atmosphere.pres_level.mean(dim="column") / 100,
            # label=f"RH: {r}",
            color=total_colors[i],
        )

        axes[1].plot(
            htgr[name][r]["lw"],
            atmosphere.pres_level.mean(dim="column") / 100,
            color=lw_colors[i],
            label=lw_label,
        )
        axes[1].plot(
            htgr[name][r]["sw"],
            atmosphere.pres_level.mean(dim="column") / 100,
            color=sw_colors[i],
            label=sw_label,
        )
for idx, shape in enumerate([quniform, qcshape, qwshape]):
    ax = axs[idx, 0]
    for idx, (r, q) in enumerate(shape.items()):
        ax.plot(
            mtf.specific_humidity_to_relative_humidity(q, pseudo.P, pseudo.T, es=es),
            pseudo.P.values / 100,
            label=f"RH: {r}",
            color=total_colors[idx],
        )


axes[1].set_xlim(-3.5, 3.5)
axes[1].invert_yaxis()
for ax in axs.flatten():
    ax.axhline(
        pseudo.P[(np.abs(pseudo.T - 273.15)).argmin()] / 100,
        color="k",
    )
for ax in axs[:, 1]:
    ax.axvline(0, color="k", linestyle="-")

# ax.set_ylim(1000, 100)
axs[0, 1].legend(loc=3)
axs[0, 0].legend(loc=2, fontsize="small")
sns.despine(offset=10)
for ax in axs[:, 0]:
    ax.set_ylabel("pressure / hPa")
axs[2, 1].set_xlabel("heating rate / K day$^{-1}$")
axs[2, 0].set_xlabel("RH / 1")

# %%

pseudo = pseudo.assign(
    stability=ph.get_stability(
        theta=pseudo.theta,
        T=pseudo.T,
    )
)

# %%
name = "cshape"
fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(10, 6))
axes[0].plot(pseudo.stability, pseudo.T, label="pseudo", color="black", linestyle="-.")
axes[0].plot(
    ph.get_stability(
        theta=mtf.theta(lvl3rad.t, P=lvl3rad.p)
        .mean("sonde")
        .sel(altitude=slice(0, 14000)),
        T=lvl3rad.t.mean("sonde").sel(altitude=slice(0, 14000)),
    ),
    lvl3rad.t.mean("sonde").sel(altitude=slice(0, 14000)),
    linestyle="-",
    label="sonde mean",
    color="black",
)

axes[1].plot(
    mtf.specific_humidity_to_relative_humidity(
        qcshape[0.9], pseudo.P, pseudo.T, es=es
    ).sel(altitude=slice(0, 16000)),
    pseudo.T.sel(altitude=slice(0, 16000)),
    label="Cshape",
    color="black",
    linestyle="--",
)
axes[1].plot(
    mtf.specific_humidity_to_relative_humidity(
        qwshape[0.9], pseudo.P, pseudo.T, es=es
    ).sel(altitude=slice(0, 16000)),
    pseudo.T.sel(altitude=slice(0, 16000)),
    label="Wshape",
    color="black",
    linestyle=":",
)
axes[1].plot(
    mtf.specific_humidity_to_relative_humidity(
        lvl3rad.q.mean("sonde").sel(altitude=slice(0, 13000)),
        lvl3rad.p.mean("sonde").sel(altitude=slice(0, 13000)),
        lvl3rad.t.mean("sonde").sel(altitude=slice(0, 13000)),
        es=es,
    ),
    lvl3rad.t.mean("sonde").sel(altitude=slice(0, 13000)),
    linestyle="-",
    label="sonde mean",
    color="black",
)
for name, ls in zip(["cshape", "wshape", "sondes"], ["--", ":", "-"]):
    axes[2].plot(
        (htgr[name][0.9]["lw"] + htgr[name][0.9]["sw"])[0:180],
        atmosphere.temp_level.isel(level=slice(0, 180), column=0),
        label="pseudo, cshape RH 0.9",
        color=total_colors[-1],
        linestyle=ls,
    )

    axes[2].plot(
        htgr[name][0.9]["lw"][0:180],
        atmosphere.temp_level.isel(level=slice(0, 180), column=0),
        color=lw_colors[-1],
        linestyle=ls,
    )
    axes[2].plot(
        htgr[name][0.9]["sw"][0:180],
        atmosphere.temp_level.isel(level=slice(0, 180), column=0),
        color=sw_colors[-1],
        linestyle=ls,
    )


axes[0].invert_yaxis()
axes[1].legend()
axes[0].legend()
axes[2].axvline(0, color="k", linestyle="-", alpha=0.5)
axes[0].set_xlabel(r"d$\theta$/dz / K km$^{-1}$")
axes[2].set_xlabel("heating rate / K day$^{-1}$")
axes[1].set_xlabel("RH / 1")
axes[0].set_ylabel("temperature / K")
for ax in axes:
    ax.axhline(273.15, color="k", alpha=0.5)
sns.despine(offset=10)

# %% csc lw vs total
rh = 0.9


fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True, sharex="col", figsize=(12, 12))

for i, (form, qs) in enumerate(zip(["cshape", "wshape"], [qcshape, qwshape])):
    rho = (
        ph.density_from_q(
            p=flxs[form][rh].pres_level.isel(column=0),
            T=flxs[form][rh].temp_level.isel(column=0),
            q=qs[rh].values,
        )
        .assign_coords(altitude=("level", pseudo.altitude.values))
        .swap_dims({"level": "altitude"})
        .drop_vars("column")
    )
    totalcool = xr.DataArray(
        -(htgr[form][rh]["lw"] + htgr[form][rh]["sw"]) / 60 / 60 / 24,
        coords={"altitude": pseudo.altitude},
    )

    csc = {"lw": {}, "sw": {}, "total": {}}
    for name in ["lw", "sw"]:
        cool = xr.DataArray(
            -htgr[form][rh][name] / 60 / 60 / 24,
            coords={"altitude": pseudo.altitude},
        )
        csc[name]["stab"] = ph.get_csc_stab(rho, pseudo.stability, cool) * 60 * 60 * 24
        csc[name]["cool"] = (
            ph.get_csc_cooling(rho, pseudo.stability, cool) * 60 * 60 * 24
        )
        csc[name]["M"] = ph.mass_flux(pseudo.stability, cool) * 60 * 60 * 24
    csc["total"]["stab"] = (
        ph.get_csc_stab(rho, pseudo.stability, totalcool) * 60 * 60 * 24
    )
    csc["total"]["cool"] = (
        ph.get_csc_cooling(rho, pseudo.stability, totalcool) * 60 * 60 * 24
    )
    csc["total"]["M"] = ph.mass_flux(pseudo.stability, totalcool) * 60 * 60 * 24

    for name, ls in zip(["lw", "sw", "total"], ["--", ":", "-"]):
        if name == "total":
            lstab = r"CSC$_{\frac{{\partial S}}{{\partial z}}}$"
            lcool = r"CSC$_{\frac{{\partial \mathcal{{H}}}}{{\partial z}}}$"
        else:
            lstab = ""
            lcool = ""
        axes[i, 0].plot(
            csc[name]["stab"].sel(altitude=slice(0, 12000)),
            pseudo.T.sel(altitude=slice(0, 12000)),
            ls=ls,
            color="C1",
            label=lstab,
        )
        axes[i, 0].plot(
            csc[name]["cool"].sel(altitude=slice(0, 12000)),
            pseudo.T.sel(altitude=slice(0, 12000)),
            ls=ls,
            color="C0",
            label=lcool,
        )
        axes[i, 0].plot(
            (csc[name]["cool"] + csc[name]["stab"]).sel(altitude=slice(0, 12000)),
            pseudo.T.sel(altitude=slice(0, 12000)),
            label=name.upper(),
            ls=ls,
            color="k",
        )
        axes[i, 1].plot
        axes[i, 1].plot(
            csc[name]["M"].sel(altitude=slice(0, 12000)),
            pseudo.T.sel(altitude=slice(0, 12000)),
            label=name.upper(),
            ls=ls,
            color="k",
        )

for ax in axes[:, 0]:
    ax.axvline(0, color="k", linestyle="-", alpha=0.5)
    ax.axhline(273.15, color="k", alpha=0.5)
    ax.set_xlim(-0.3, 0.3)

    ax.set_ylim(295, 250)
axes[0, 1].set_xlim(-7e-4, 7e-4)
for ax in axes[:, 1]:
    ax.axvline(0, color="k", linestyle="-", alpha=0.5)
    ax.axhline(273.15, color="k", alpha=0.5)
ax.legend()
for ax in axes[:, 0]:
    ax.set_ylabel("Temperature / K")
axes[0, 0].legend()


axes[1, 0].set_xlabel("CSC(?) / s$^{-1}$")
axes[1, 1].set_xlabel("M (?) / kg m$^{-2}$ day$^{-1}$")
sns.despine(offset=10)
fig.savefig("../../../plots/csc.pdf")
# ax.invert_yaxis()
