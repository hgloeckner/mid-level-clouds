# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from xhistogram.xarray import histogram
import myutils.open_datasets as od
import myutils.physics_helper as ph
import myutils.data_helper as dh
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.constants as mtc
import moist_thermodynamics.saturation_vapor_pressures as svp
from pyrte_rrtmgp.rrtmgp import GasOptics
from pyrte_rrtmgp.rrtmgp_data_files import GasOpticsFiles

from radiation_for_sondes.rrtmg import angles
import radiation_for_sondes.rrtmg.rad_helper as rad

# %%
rrtmg_fluxes = xr.open_dataset(
    "/scratch/m/m301046/rrtmgp_sonde_fluxes.zarr", engine="zarr"
)

beach = (
    od.open_dropsondes(od.get_cids()["dropsondes"])
    .pipe(dh.interpolate_gaps)
    .pipe(dh.extrapolate_sfc)
)

# %%

wct = xr.open_dataset(
    "/scratch/m/m301046/wales_radar_cloud_top_max.zarr", engine="zarr"
)

beach = beach.assign(cth=wct["cloud-top"].sel(time=beach.launch_time, method="nearest"))
# %%
rrtmg_fluxes = xr.open_dataset(
    "/scratch/m/m301046/rrtmgp_sonde_fluxes.zarr", engine="zarr"
)
rrtmg_fluxes = rrtmg_fluxes.assign(
    lw_htgr=xr.apply_ufunc(
        ph.calc_heating_rate_from_flx,
        rrtmg_fluxes.lw_flux_up,
        rrtmg_fluxes.lw_flux_down,
        rrtmg_fluxes.pres_level,
        input_core_dims=[["level"], ["level"], ["level"]],
        output_core_dims=[["level"]],
        vectorize=True,
    ),
    sw_htgr=xr.apply_ufunc(
        ph.calc_heating_rate_from_flx,
        rrtmg_fluxes.sw_flux_up,
        rrtmg_fluxes.sw_flux_down,
        rrtmg_fluxes.pres_level,
        input_core_dims=[["level"], ["level"], ["level"]],
        output_core_dims=[["level"]],
        vectorize=True,
    ),
)
# %%

rrtflx = rrtmg_fluxes.sel(level=slice(None, 130))
hist_flx = histogram(
    rrtflx.temp_level,
    bins=[np.arange(220, 305, 1)],
    dim=["level"],
    weights=rrtflx.lw_htgr,
) / histogram(
    rrtflx.temp_level,
    bins=[np.arange(220, 305, 1)],
    dim=["level"],
)
hist_flx = hist_flx.to_dataset(name="lw_htgr").assign(
    sw_htgr=(
        histogram(
            rrtflx.temp_level,
            bins=[np.arange(220, 305, 1)],
            dim=["level"],
            weights=rrtflx.sw_htgr,
        )
        / histogram(
            rrtflx.temp_level,
            bins=[np.arange(220, 305, 1)],
            dim=["level"],
        )
    ),
    sonde_id=rrtflx.sonde_id,
    launch_time=rrtflx.launch_time,
    launch_lat=rrtflx.launch_lat,
    launch_lon=rrtflx.launch_lon,
)

# %%
no_sids = [
    sid
    for sid in beach.where(np.isnan(beach.cth), drop=True).sonde_id.values
    if sid in rrtmg_fluxes.sonde_id
]
low_sids = [
    sid
    for sid in beach.where(beach.cth < 4000, drop=True).sonde_id.values
    if sid in rrtmg_fluxes.sonde_id
]
mid_sids = [
    sid
    for sid in beach.where(
        (beach.cth >= 4000) & (beach.cth < 8000), drop=True
    ).sonde_id.values
    if sid in rrtmg_fluxes.sonde_id
]
high_sids = [
    sid
    for sid in beach.where(beach.cth >= 8000, drop=True).sonde_id.values
    if sid in rrtmg_fluxes.sonde_id
]

sids = {
    "Clear Sky": no_sids,
    "CTH < 4 km": low_sids,
    "CTH 4-8 km": mid_sids,
    "CTH > 8 km": high_sids,
}

# %%
regions = {
    "east": dh.east,
    "west": dh.west,
    "north": dh.north,
}
mlbeach = beach.where((beach.cth >= 4000) & (beach.cth < 8000), drop=True)
for name, region in regions.items():
    reg_beach = dh.sel_sub_domain(mlbeach, region)
    mean_cth = reg_beach.cth.mean("sonde")
    print(f"{name} mean CTH", mean_cth.values)
    print(
        f"{name} mean Beach CTT",
        reg_beach.interp(altitude=mean_cth).ta.mean("sonde").values,
        "$\\pm$",
        reg_beach.interp(altitude=mean_cth).ta.std("sonde").values,
    )
    print(
        "10 and 90 percentiles:",
        reg_beach.interp(altitude=mean_cth).ta.quantile(0.1, dim="sonde").values,
        reg_beach.interp(altitude=mean_cth).ta.quantile(0.9, dim="sonde").values,
    )
reg_beach = mlbeach
mean_cth = reg_beach.cth.mean("sonde")
print("total mean CTH", mean_cth.values)
print(
    "total mean Beach CTT",
    reg_beach.interp(altitude=mean_cth).ta.mean("sonde").values,
    "$\\pm$",
    reg_beach.interp(altitude=mean_cth).ta.std("sonde").values,
)
print(
    "10 and 90 percentiles:",
    reg_beach.interp(altitude=mean_cth).ta.quantile(0.1, dim="sonde").values,
    reg_beach.interp(altitude=mean_cth).ta.quantile(0.9, dim="sonde").values,
)

# %%

# %%


fig, ax = plt.subplots(figsize=(6, 4))
for ls, (label, ids) in zip(["-", "--", "-.", ":"], sids.items()):
    pltds = hist_flx.swap_dims({"column": "sonde_id"}).sel(
        sonde_id=ids
    )  # .sel(level=slice(None, 130))

    (pltds.mean("sonde_id").lw_htgr * 3600 * 24).plot(
        y="temp_level_bin",
        ax=ax,
        color="blue",
        linestyle=ls,
    )
    (pltds.mean("sonde_id").sw_htgr * 3600 * 24).plot(
        y="temp_level_bin",
        ax=ax,
        color="red",
        linestyle=ls,
    )
    (
        pltds.mean("sonde_id").sw_htgr * 3600 * 24
        + pltds.mean("sonde_id").lw_htgr * 3600 * 24
    ).plot(
        y="temp_level_bin",
        ax=ax,
        color="k",
        linestyle=ls,
        label=label,
    )
ax.invert_yaxis()
ax.axhline(273.15, color="grey", linestyle="--")
ax.set_xlim(-4, 3)
ax.axvline(0, color="grey", linestyle="--")
sns.despine(offset=10)
ax.set_xlabel("Heating Rate / K day$^{-1}$")
ax.set_ylabel("Temperature / K")
ax.legend()

# %%


def calc_cs_convergence(ds, ta_var="ta"):
    res = {}
    res["stab"] = ph.get_stability(ds.theta, ds[ta_var]).rolling(altitude=5).mean()
    res["rho"] = ph.density_from_q(ds.p, ds[ta_var], ds.q)
    res["csc_stab"] = ph.get_csc_stab(
        res["rho"], res["stab"], -ds.lw_htgr.rolling(altitude=5, center=True).mean()
    )
    res["csc_cool"] = ph.get_csc_cooling(
        res["rho"], res["stab"], -ds.lw_htgr.rolling(altitude=5, center=True).mean()
    )
    return res


# %%
def get_data(key):
    ds = (
        rrtmg_fluxes.swap_dims({"column": "sonde_id"})
        .sel(sonde_id=sids[key])
        .mean("sonde_id")
        .swap_dims({"level": "pres_level"})
    )
    beachds = (
        beach.swap_dims({"sonde": "sonde_id"})
        .sel(sonde_id=sids[key])
        .mean("sonde_id")
        .swap_dims({"altitude": "p"})
        .interp(
            p=ds.pres_level,
            kwargs={"fill_value": "extrapolate"},
        )
    )
    ds = (
        ds.assign(
            altitude=beachds.altitude,
            q=beachds.q,
            theta=beachds.theta,
        )
        .swap_dims({"pres_level": "altitude"})
        .rename(temp_level="ta")
    )

    return ds


data = {key: get_data(key) for key in sids.keys()}
# %%

fig, ax = plt.subplots(figsize=(6, 4))
for key, item in data.items():
    (
        data[key].theta.interp(altitude=data["Clear Sky"].altitude)
        - data["Clear Sky"].theta
    ).plot(y="altitude")

ax.legend()
ax.set_ylim(0, 15000)
ax.set_xlim(-2, 2)
# %%

# lstab = r"CSC$_{\frac{{\partial S}}{{\partial z}}}$"
# lcool = r"CSC$_{\frac{{\partial \mathcal{{H}}}}{{\partial z}}}$"
cm = 1 / 2.54
cw = 20 * cm
fig, axes = plt.subplots(
    ncols=3,
    nrows=2,
    figsize=(cw, cw),
    sharey=True,
)

for idx, (ls, key) in enumerate(
    zip(["--", ":", "-"], ["Clear Sky", "CTH < 4 km", "CTH 4-8 km"])
):
    ds = data[key]

    csc = calc_cs_convergence(ds)

    axes[0, 0].plot(
        -ds.lw_htgr.isel(altitude=slice(0, 130)) * 3600 * 24,
        ds.ta.isel(altitude=slice(0, 130)),
        color="k",
        ls=ls,
        label=key,
    )

    axes[0, 1].plot(
        csc["stab"].isel(altitude=slice(0, 130)),
        ds.ta.isel(altitude=slice(0, 130)),
        color="k",
        ls=ls,
    )

    axes[1, idx].plot(
        csc["csc_stab"] * 3600 * 24,
        ds.ta,
        color="C1",
    )
    axes[1, idx].plot(
        csc["csc_cool"] * 3600 * 24,
        ds.ta,
        color="C0",
    )
    axes[1, idx].plot(
        (csc["csc_cool"] + csc["csc_stab"]) * 3600 * 24, ds.ta, color="k", label=key
    )
    axes[0, 2].plot(
        (csc["csc_cool"] + csc["csc_stab"]) * 3600 * 24, ds.ta, color="k", ls=ls
    )
axes[0, 0].legend()
axes[0, 0].set_xlabel("LW Heating Rate / K day$^{-1}$")
axes[0, 1].set_xlabel("Stability / K m$^{-1}$")
axes[0, 0].set_ylabel("Temperature / K")
axes[1, 0].set_ylabel("Temperature / K")
axes[0, 1].invert_yaxis()
axes[0, 1].set_ylim(280, 260)
axes[0, 1].set_xlim(2.5, 5)
axes[0, 0].set_xlim(1, 3.5)
axes[0, 2].set_xlabel("LW Convergence / s$^{-1}$")
axes[0, 2].set_xlim(-0.3, 0.2)
axes[0, 2].axvline(0, color="grey", linestyle="-", alpha=0.5)
for ax in axes[1, :]:
    ax.axvline(0, color="grey", linestyle="-", alpha=0.5)
    ax.set_xlabel("LW Convergence / s$^{-1}$")
    ax.set_xlim(-0.4, 0.3)
    ax.legend()
for ax in axes.flatten():
    ax.axhline(273.15, color="grey", linestyle="-", alpha=0.5)
sns.despine(offset=10)
fig.tight_layout()
fig.savefig("/scratch/m/m301046/csc_lw_cth.pdf")
# %%
"""
todo:
  - sw for complete day for real RH and T
  - moist adiabat profile with corresponding real RH profiles -> RRTMG
  - real T with idealized RH (E and C) -> RRTMG
"""


# %%
beachdata = {}
raddata = {}
radbeach = xr.open_dataset(
    "/work/mh0066/m301046/ml_clouds/sondes_for_radiation.nc"
).swap_dims({"sonde": "sonde_id"})

beachdata["CTH < 4 km"] = beach.swap_dims({"sonde": "sonde_id"}).sel(
    sonde_id=[sid for sid in low_sids] + [sid for sid in no_sids]
)
data["CTH < 4 km"] = rrtmg_fluxes.swap_dims({"column": "sonde_id"}).sel(
    sonde_id=[sid for sid in low_sids] + [sid for sid in no_sids]
)

for key in ["CTH > 8 km", "CTH 4-8 km"]:
    beachdata[key] = beach.swap_dims({"sonde": "sonde_id"}).sel(sonde_id=sids[key])
    raddata[key] = rrtmg_fluxes.swap_dims({"column": "sonde_id"}).sel(
        sonde_id=sids[key]
    )

# %%

# %% idealized radiation


pseudos = {}
zlcl = {}
for key in ["CTH < 4 km", "CTH 4-8 km", "CTH > 8 km"]:
    Psfc = beachdata[key].p.mean(dim="sonde_id").sel(altitude=0).values
    P = np.arange(Psfc, 4000.0, -500)
    sfcT = beachdata[key].ta.mean(dim="sonde_id").sel(altitude=0).values
    qsfc = (
        beachdata[key].q.mean(dim="sonde_id").sel(altitude=0).values
    )  # 9182267570514704

    pseudo = ph.make_sounding_from_adiabat(
        P, sfcT, qsfc, thx=mtf.theta_e_bolton, Tmin=195
    )
    pseudo = xr.concat(
        [
            pseudo,
            radbeach[["t", "p"]]
            .mean("sonde_id")
            .rename({"t": "T", "p": "P"})
            .sel(altitude=slice(pseudo.altitude.max().values, None)),
        ],
        dim="altitude",
        compat="no_conflicts",
    )
    pseudo = pseudo.where(pseudo.T > 195).interpolate_na("altitude", method="akima")
    pseudos[key] = pseudo
    plcl = mtf.plcl_bolton(T=sfcT, P=Psfc, qt=qsfc)
    zlcl[key] = mtf.zlcl(plcl, T=sfcT, P=Psfc, qt=qsfc, z=0)
# %%

# %%
qkwargs = {}
es = mtf.make_es_mxd(
    svp.liq_wagner_pruss, svp.ice_wagner_etal
)  # svp.liq_wagner_pruss #
qkwargs["CTH < 4 km"] = {
    "rhmid": 0.4,
    "rhlcl": 0.9,
    "rhtoa": 0.3,
    "Tmin": 270,
    "es": es,
    "factor": 0.42,
    "lowlim": 286,
    "highlim": 260,
}
qkwargs["CTH 4-8 km"] = {
    "rhmid": 0.32,
    "rhlcl": 0.9,
    "rhtoa": 0.4,
    "Tmin": 250,
    "es": es,
    "factor": 0.67,
    "lowlim": 285,
    "highlim": 255,
}
qkwargs["CTH > 8 km"] = {
    "rhmid": 0.42,
    "rhlcl": 0.9,
    "rhtoa": 0.7,
    "Tmin": 250,
    "es": es,
    "factor": 0.52,
    "lowlim": 285,
    "highlim": 262,
}
fig, ax = plt.subplots(figsize=(6, 4))
for key, ls in zip(["CTH < 4 km", "CTH 4-8 km", "CTH > 8 km"], ["--", ":", "-"]):
    ds = beachdata[key]
    ax.plot(
        ds.rh.mean("sonde_id"),
        ds.ta.mean("sonde_id"),
        linestyle=ls,
        label=key,
    )
    qclow = rad.wshape_humidity(pseudos[key], zlcl=zlcl[key], **qkwargs[key])
    ax.plot(
        mtf.specific_humidity_to_relative_humidity(
            qclow,
            pseudos[key].P,
            pseudos[key].T,
            es=es,
        ),
        pseudos[key].T,
    )
ax.set_xlim(0, 1)
ax.invert_yaxis()
# ax.set_ylim(305, 270)
# ax.set_ylim(0, 20000)


# %%
qc = {"real": {}, "pseudo": {}}
qe = {"real": {}, "pseudo": {}}
qr = {"real": {}, "pseudo": {}}
alts = {"CTH < 4 km": 10000, "CTH 4-8 km": 12500, "CTH > 8 km": 12500}
for key in ["CTH < 4 km", "CTH 4-8 km", "CTH > 8 km"]:
    histreal = histogram(
        beachdata[key].ta,
        bins=[np.arange(220, 305, 0.5)],
        dim=["altitude"],
        weights=beachdata[key].rh,
    ) / histogram(
        beachdata[key].ta,
        bins=[np.arange(220, 305, 0.5)],
        dim=["altitude"],
    )
    qcreal = mtf.relative_humidity_to_specific_humidity(
        histreal.mean("sonde_id").interp(
            ta_bin=pseudos[key].T, kwargs={"fill_value": "extrapolate"}
        ),
        pseudos[key].P,
        pseudos[key].T,
        es=es,
    )
    qc["pseudo"][key] = rad.cshape_humidity(
        pseudos[key], zlcl=zlcl[key], **qkwargs[key]
    )
    qe["pseudo"][key] = rad.wshape_humidity(
        pseudos[key], zlcl=zlcl[key], **qkwargs[key]
    )
    qr["pseudo"][key] = xr.concat(
        [
            qcreal.sel(altitude=slice(None, alts[key])).drop(["ta_bin"]),
            qc["pseudo"][key].sel(altitude=slice(alts[key], None)),
        ],
        dim="altitude",
    )
# %% RH "fit" control

sns.set_palette("Paired")

fig, ax = plt.subplots(figsize=(6, 4))
for k in ["CTH < 4 km", "CTH 4-8 km", "CTH > 8 km"]:
    ax.plot(
        mtf.specific_humidity_to_relative_humidity(
            qe["pseudo"][k],
            pseudos[k].P,
            pseudos[k].T,
            es=es,
        ),
        pseudos[k].altitude,
        linestyle=ls,
    )
    ax.plot(
        mtf.specific_humidity_to_relative_humidity(
            qr["pseudo"][k],
            pseudos[k].P,
            pseudos[k].T,
            es=es,
        ),
        pseudos[k].altitude,
        linestyle=ls,
        label=k,
    )

ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 20000)
# ax.invert_yaxis()
# ax.set_ylim(305, 220)
sns.despine()


# %%
def calc_radiation(qc, qe, qr, pseudos):
    res = {key: {} for key in pseudos.keys()}
    for key in res.keys():
        pseudo = pseudos[key]
        for name, q in zip(["C shape", "E shape", "Real"], [qc[key], qe[key], qr[key]]):
            print(name)
            mu0 = angles.get_mu_day(
                np.datetime64("2024-08-30T00:00:00"), lat=0, lon=-30
            )
            atmosphere = rad.make_atmosphere(
                pseudo.P.values.reshape(1, pseudo.P.shape[0]),
                pseudo.T.values.reshape(1, pseudo.P.shape[0]),
                ph.specific_humidity2vmr(q).values.reshape(1, pseudo.P.shape[0]),
                o3=radbeach.O3.interp(altitude=pseudo.altitude).values,
            )

            assert not np.any(
                [np.any(np.isnan(atmosphere[var])) for var in atmosphere.variables]
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

            res[key][name] = xr.merge(
                [
                    lw_fluxes,
                    xr.concat(sw_fluxes, dim="mu0"),
                    atmosphere,
                ]  # mean(dim="mu0")
            )
            res[key][name] = (
                res[key][name]
                .assign(
                    lw_htgr=xr.apply_ufunc(
                        ph.calc_heating_rate_from_flx,
                        res[key][name].lw_flux_up,
                        res[key][name].lw_flux_down,
                        res[key][name].pres_level,
                        input_core_dims=[["level"], ["level"], ["level"]],
                        output_core_dims=[["level"]],
                        vectorize=True,
                    ),
                    sw_htgr=xr.apply_ufunc(
                        ph.calc_heating_rate_from_flx,
                        res[key][name].sw_flux_up,
                        res[key][name].sw_flux_down,
                        res[key][name].pres_level,
                        input_core_dims=[["level"], ["level"], ["level"]],
                        output_core_dims=[["level"]],
                        vectorize=True,
                    ),
                    theta=mtf.theta(
                        res[key][name].temp_level,
                        res[key][name].pres_level,
                    ),
                    altitude=("level", pseudo.altitude.values),
                    q=("level", q.values),
                )
                .rename(
                    temp_level="ta",
                    pres_level="p",
                )
                .swap_dims({"level": "altitude"})
            )
    return res


# %%
res_pseudo = calc_radiation(qc["pseudo"], qe["pseudo"], qr["pseudo"], pseudos)


# %%
def calc_cs_convergence_sw(ds, ta_var="ta"):
    res = {}
    res["stab"] = ph.get_stability(ds.theta, ds[ta_var]).rolling(altitude=5).mean()
    res["rho"] = ph.density_from_q(ds.p, ds[ta_var], ds.q)
    htgr = ds.lw_htgr + ds.sw_htgr.mean("mu0")
    res["csc_stab"] = ph.get_csc_stab(
        res["rho"], res["stab"], -htgr.rolling(altitude=5, center=True).mean()
    )
    res["csc_cool"] = ph.get_csc_cooling(
        res["rho"], res["stab"], -htgr.rolling(altitude=5, center=True).mean()
    )
    res["lw_csc_stab"] = ph.get_csc_stab(
        res["rho"], res["stab"], -ds.lw_htgr.rolling(altitude=5, center=True).mean()
    )
    res["lw_csc_cool"] = ph.get_csc_cooling(
        res["rho"], res["stab"], -ds.lw_htgr.rolling(altitude=5, center=True).mean()
    )
    res["rho"] = ph.density_from_q(ds.p, ds[ta_var], ds.q)
    res["mass_flux"] = ph.mass_flux(
        res["rho"], res["stab"], -htgr.rolling(altitude=5, center=True).mean()
    )
    res["lw_mass_flux"] = ph.mass_flux(
        res["rho"], res["stab"], -ds.lw_htgr.rolling(altitude=5, center=True).mean()
    )
    return res


# %%
def get_cooling_from_mass_flux(mass_flux, stability, rho):
    return -mass_flux * stability / rho


res_key = "CTH > 8 km"
sw = False
if sw:
    mass_var = "mass_flux"
else:
    mass_var = "lw_mass_flux"
sns.set_palette("tab10")
fig, axes = plt.subplots(figsize=(12, 4), ncols=3)
for key, ls in zip(["C shape", "E shape", "Real"], ["--", ":", "-"]):
    ds = res_pseudo[res_key][key].sel(column=0, altitude=slice(None, 12000))

    csc = calc_cs_convergence_sw(ds)
    uniform_mass = np.full(
        ds.altitude.shape, csc[mass_var].sel(altitude=slice(1000, None)).mean().values
    )
    idealized_cooling = get_cooling_from_mass_flux(
        uniform_mass,
        csc["stab"],
        csc["rho"],
    )

    axes[0].plot(
        uniform_mass * 3600 * 24,
        ds.altitude,
        color="C0",
        linestyle=ls,
    )
    if sw:
        htgr = (ds.lw_htgr + ds.sw_htgr.mean("mu0")) * 3600 * 24
    else:
        htgr = ds.lw_htgr * 3600 * 24

    (csc[mass_var].sel(altitude=slice(1000, None)) * 3600 * 24).plot(
        y="altitude",
        ax=axes[0],
        color="C1",
        linestyle=ls,
    )

    htgr.plot(  #
        y="altitude",
        ax=axes[1],
        label=f"{key} heating",
        color="C1",
        linestyle=ls,
    )

    (idealized_cooling * 3600 * 24).plot(
        y="altitude",
        ax=axes[1],
        color="C0",
        linestyle=ls,
    )

    (htgr - idealized_cooling * 3600 * 24).plot(  # + ds.sw_htgr.mean("mu0")
        y="altitude",
        ax=axes[2],
        label="residual",
        color="k",
        linestyle=ls,
    )

    mean_htg = (
        (htgr / 3600 / 24 - idealized_cooling)
        .sel(altitude=slice(1000, 6000))
        .mean()
        .values
    )
    dp = (
        ds.p.interp(altitude=[1000, 6000])[0] - ds.p.interp(altitude=[1000, 6000])[1]
    ) / 100
    print(
        "sw:", sw, res_key, key, "energy 1-6km", (dp.values * 10) * mean_htg * mtc.cpv
    )  # J s-1


axes[0].set_xlabel("Mass Flux / kg m$^{-2}$ day$^{-1}$")
axes[1].set_xlabel("$\\mathcal{{H}}$ / K day$^{-1}$")
axes[2].set_xlabel("$\\Delta\\mathcal{{H}}$/ K day$^{-1}$")
for ax in axes.flatten():
    ax.set_ylabel("")
    ax.set_title("")

axes[0].set_ylabel("altitude / m")
axes[1].legend()
axes[2].axvline(0, color="grey", linestyle="-", alpha=0.5)
sns.despine(offset={"left": 10})
# %%
print(
    10 * mtc.melting_enthalpy_stp / (60 * 60 * 24)  # J s-1
)

# %%
# Todo: real temperature structure
realmeans = {}
zlcl = {}
for key in ["CTH < 4 km", "CTH 4-8 km", "CTH > 8 km"]:
    ds = beachdata[key].mean(dim="sonde_id")[["ta", "p"]].rename({"ta": "T", "p": "P"})

    ds = xr.concat(
        [
            ds,
            radbeach[["t", "p"]]
            .mean("sonde_id")
            .rename({"t": "T", "p": "P"})
            .sel(altitude=slice(ds.altitude.max().values, None)),
        ],
        dim="altitude",
        compat="no_conflicts",
    )
    realmeans[key] = ds.where(ds.T > 195).interpolate_na("altitude", method="akima")

    zlcl[key] = 500
# %%
alts = {"CTH < 4 km": 10001, "CTH 4-8 km": 11501, "CTH > 8 km": 12001}
for key in ["CTH < 4 km", "CTH 4-8 km", "CTH > 8 km"]:
    histreal = histogram(
        beachdata[key].ta,
        bins=[np.arange(220, 305, 0.5)],
        dim=["altitude"],
        weights=beachdata[key].rh,
    ) / histogram(
        beachdata[key].ta,
        bins=[np.arange(220, 305, 0.5)],
        dim=["altitude"],
    )
    qcreal = mtf.relative_humidity_to_specific_humidity(
        histreal.mean("sonde_id").interp(
            ta_bin=realmeans[key].T, kwargs={"fill_value": "extrapolate"}
        ),
        realmeans[key].P,
        realmeans[key].T,
        es=es,
    )
    qc["real"][key] = rad.cshape_humidity(
        realmeans[key], zlcl=zlcl[key], **qkwargs[key]
    )
    qe["real"][key] = rad.wshape_humidity(
        realmeans[key], zlcl=zlcl[key], **qkwargs[key]
    )
    qr["real"][key] = xr.concat(
        [
            qcreal.sel(altitude=slice(None, alts[key])).drop(["ta_bin"]),
            qc["real"][key].sel(altitude=slice(alts[key], None)),
        ],
        dim="altitude",
    )


fig, ax = plt.subplots(figsize=(6, 4))
for key, ls in zip(["CTH < 4 km", "CTH 4-8 km", "CTH > 8 km"], ["--", ":", "-"]):
    ds = beachdata[key]
    ax.plot(
        ds.rh.mean("sonde_id"),
        ds.ta.mean("sonde_id"),
        linestyle=ls,
        label=key,
    )
    ax.plot(
        mtf.specific_humidity_to_relative_humidity(
            qr["real"][key],
            realmeans[key].P,
            realmeans[key].T,
            es=es,
        ),
        realmeans[key].T,
    )
ax.set_xlim(0, 1)
ax.invert_yaxis()
# ax.set_ylim(305, 270)
# ax.set_ylim(0, 20000)
# %%


# %%
res_real = calc_radiation(qc["real"], qe["real"], qr["real"], realmeans)
# %%


# %%
def get_cooling_from_mass_flux(mass_flux, stability, rho):
    return -mass_flux * stability / rho


res_key = "CTH < 4 km"
sw = True
if sw:
    mass_var = "mass_flux"
    label = "including SW"
    mass_var_comp = "lw_mass_flux"
    label_comp = "LW only"
else:
    mass_var = "lw_mass_flux"
    label = "LW only"
    mass_var_comp = "mass_flux"
    label_comp = "including SW"
sns.set_palette("tab10")
fig, axs = plt.subplots(figsize=(12, 8), ncols=3, nrows=2, sharey=True, sharex="col")
for idx, (res, roll) in enumerate(zip([res_pseudo, res_real], [1, 50])):
    axes = axs[idx]
    for key, ls in zip(["C shape", "E shape", "Real"], ["--", ":", "-"]):
        ds = res[res_key][key].sel(column=0, altitude=slice(None, 12000))

        csc = calc_cs_convergence_sw(ds)
        csc["stab"] = csc["stab"].rolling(altitude=roll, center=True).mean()
        csc[mass_var] = csc[mass_var].rolling(altitude=roll, center=True).mean()
        csc[mass_var_comp] = (
            csc[mass_var_comp].rolling(altitude=roll, center=True).mean()
        )
        uniform_mass = np.full(
            ds.altitude.shape,
            csc[mass_var].sel(altitude=slice(1000, None)).mean().values,
        )
        idealized_cooling = get_cooling_from_mass_flux(
            uniform_mass,
            csc["stab"],
            csc["rho"],
        )
        idealized_cooling_comp = get_cooling_from_mass_flux(
            np.full(
                ds.altitude.shape,
                csc[mass_var_comp].sel(altitude=slice(1000, None)).mean().values,
            ),
            csc["stab"],
            csc["rho"],
        )

        axes[0].plot(
            uniform_mass * 3600 * 24,
            ds.altitude,
            color="C0",
            linestyle=ls,
            label=key,
        )
        if sw:
            htgr = (
                ((ds.lw_htgr + ds.sw_htgr.mean("mu0")) * 3600 * 24)
                .rolling(altitude=roll, center=True)
                .mean()
            )
            htgr_comp = (
                (ds.lw_htgr * 3600 * 24).rolling(altitude=roll, center=True).mean()
            )
        else:
            htgr = (ds.lw_htgr * 3600 * 24).rolling(altitude=roll, center=True).mean()
            htgr_comp = (
                ((ds.lw_htgr + ds.sw_htgr.mean("mu0")) * 3600 * 24)
                .rolling(altitude=roll, center=True)
                .mean()
            )

        (csc[mass_var].sel(altitude=slice(1000, None)) * 3600 * 24).plot(
            y="altitude",
            ax=axes[0],
            color="C1",
            linestyle=ls,
        )

        htgr.plot(  #
            y="altitude",
            ax=axes[1],
            # label=f"{key} heating",
            color="C1",
            linestyle=ls,
        )

        (idealized_cooling * 3600 * 24).plot(
            y="altitude",
            ax=axes[1],
            color="C0",
            linestyle=ls,
        )

        if key == "Real":
            reslabel = label
            reslabel_comp = label_comp
        else:
            reslabel = None
            reslabel_comp = None
        (htgr - idealized_cooling * 3600 * 24).plot(  # + ds.sw_htgr.mean("mu0")
            y="altitude",
            ax=axes[2],
            label=reslabel,
            color="k",
            linestyle=ls,
        )
        (
            htgr_comp - idealized_cooling_comp * 3600 * 24
        ).plot(  # + ds.sw_htgr.mean("mu0")
            y="altitude",
            ax=axes[2],
            color="grey",
            linestyle=ls,
            label=reslabel_comp,
        )

        mean_htg = (
            (htgr / 3600 / 24 - idealized_cooling)
            .sel(altitude=slice(1000, 6000))
            .mean()
            .values
        )
        dp = (
            ds.p.interp(altitude=[1000, 6000])[0]
            - ds.p.interp(altitude=[1000, 6000])[1]
        ) / 100
        print(
            "sw:",
            sw,
            res_key,
            key,
            "energy 1-6km",
            (dp.values * 10) * mean_htg * mtc.cpv,
        )  # J s-1

    dH = (
        -40
        / 10
        / mtc.cpv
        / ((ds.p.interp(altitude=2500) - ds.p.interp(altitude=5000)) / 100)
    )
    axes[1].axvline(
        dH * 3600 * 24,
        1.7 / 10,
        4.4 / 10,
        color="grey",
    )

axes[0].set_xlabel("Mass Flux / kg m$^{-2}$ day$^{-1}$")
axes[1].set_xlabel("$\\mathcal{{H}}$ / K day$^{-1}$")
axes[1].set_xlim(-3.5, -0.5)
axes[2].set_xlim(-1, 1)
axes[2].set_xlabel("$\\Delta\\mathcal{{H}}$/ K day$^{-1}$")
axes[2].legend()
for ax in axs.flatten():
    ax.set_ylabel("")
    ax.set_title("")
for ax in axs[:, 0]:
    ax.set_ylabel("altitude / m")
    ax.set_ylim(1000, 10000)
axs[0, 0].legend()
# axs[0, 1].legend()
for ax in axs[:, -1]:
    ax.axvline(0, color="grey", linestyle="-", alpha=0.5)
sns.despine(offset={"left": 10})
fig.savefig("/scratch/m/m301046/mlclouds/plots/mflux_mxd.pdf")

# %%
# %%
res = res_real
res_key = "CTH < 4 km"

for key, ls in zip(["C shape", "E shape", "Real"], ["--", ":", "-"]):
    ds = res[res_key][key].sel(column=0, altitude=slice(None, 12000))
    dH = (
        -40 / 10 / mtc.cpv / ((ds.p.sel(altitude=2500) - ds.p.sel(altitude=5000)) * 100)
    )
    csc = calc_cs_convergence_sw(ds)
    csc["stab"] = csc["stab"].rolling(altitude=roll, center=True).mean()
    csc[mass_var] = csc[mass_var].rolling(altitude=roll, center=True).mean()

    htgr = (
        (ds.lw_htgr + ds.sw_htgr.mean("mu0")).rolling(altitude=roll, center=True).mean()
    )
    mflux = ph.mass_flux(csc["rho"], csc["stab"], (htgr - dH.values))

    htgr.plot(y="altitude")
    (htgr - dH.values).plot(y="altitude")
    htgr.loc[dict(altitude=slice(2500, 5000))] = (
        htgr.loc[dict(altitude=slice(2500, 5000))] - dH.values
    )
    htgr.plot(y="altitude")

sns.despine(offset={"left": 10})
