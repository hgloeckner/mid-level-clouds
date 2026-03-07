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
import myutils.moist_adiabats as ma
from radiation_for_sondes.rrtmg import angles
import radiation_for_sondes.rrtmg.rad_helper as rad

es = mtf.make_es_mxd(svp.liq_wagner_pruss, svp.ice_wagner_etal)

levante = True

if levante:
    file_path = "/scratch/m/m301046/"
    cth_path = "/work/mh0066/m301046/ml_clouds/sondes_for_radiation.nc"

else:
    file_path = "/Users/helene/Documents/code/mid_level_clouds/plots/"
    cth_path = file_path + "sondes_for_radiation.nc"


# %%

rrtmg_fluxes = xr.open_dataset(file_path + "rrtmgp_sonde_fluxes.zarr", engine="zarr")

beach = (
    od.open_dropsondes(od.get_cids()["dropsondes"])
    .pipe(dh.interpolate_gaps)
    .pipe(dh.extrapolate_sfc)
)

wct = xr.open_dataset(file_path + "wales_radar_cloud_top_max.zarr", engine="zarr")

beach = beach.assign(cth=wct["cloud-top"].sel(time=beach.launch_time, method="nearest"))

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
    "CTH < 4 km": low_sids + no_sids,
    "CTH 4-8 km": mid_sids,
    "CTH > 8 km": high_sids,
}


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
beachdata = {}
raddata = {}
radbeach = xr.open_dataset(cth_path, engine="netcdf4").swap_dims({"sonde": "sonde_id"})

for key in ["CTH < 4 km", "CTH > 8 km", "CTH 4-8 km"]:
    beachdata[key] = beach.swap_dims({"sonde": "sonde_id"}).sel(sonde_id=sids[key])
    raddata[key] = rrtmg_fluxes.swap_dims({"column": "sonde_id"}).sel(
        sonde_id=sids[key]
    )

# %% idealized radiation
adiabat_fct = {
    "pseudo": ma.pseudo_adiabat,
    "reversible": ma.reversible_adiabat,
}
alt = 0
adiabat_dict = {}
for key in ["CTH < 4 km", "CTH 4-8 km", "CTH > 8 km"]:
    Psfc = beachdata[key].p.mean(dim="sonde_id").sel(altitude=0).values
    P = np.arange(Psfc, 4000.0, -500)
    sfcT = beachdata[key].ta.mean(dim="sonde_id").sel(altitude=0).values
    qsfc = beachdata[key].q.mean(dim="sonde_id").sel(altitude=0).values
    adiabat_list = []
    for name, ma_fct in adiabat_fct.items():
        adiabat = ma.make_sounding_from_adiabat(
            ma_fct=ma_fct,
            P=P,
            Tsfc=sfcT,
            qsfc=qsfc,
            Tmin=195,
        )
        adiabat = xr.concat(
            [
                adiabat,
                radbeach[["t", "p"]]
                .mean("sonde_id")
                .rename({"t": "T", "p": "P"})
                .sel(altitude=slice(adiabat.altitude.max().values, None)),
            ],
            dim="altitude",
            compat="no_conflicts",
        )
        adiabat = (
            adiabat.where(adiabat.T > 195)
            .interpolate_na("altitude", method="akima")
            .assign(adiabat=name)
        )
        adiabat_list.append(adiabat)
        if np.all(alt == 0):
            alt = adiabat_list[0].altitude
    adiabat_list = [adiabat.interp(altitude=alt) for adiabat in adiabat_list]
    adiabat_dict[key] = xr.concat(adiabat_list, dim="adiabat")

adiabat_ds = xr.concat(
    [adiabat_dict[key].assign(cth=key) for key in adiabat_dict.keys()],
    dim="cth",
)
# %%
for adiabat in ["pseudo", "reversible"]:  # adiabat_ds.adiabat.values:
    # adiabat_ds.sel(adiabat=adiabat, cth="CTH < 4 km").T.plot(label=adiabat)
    adiabat_ds.sel(cth="CTH 4-8 km", adiabat=adiabat).T.plot()

plt.legend()
# %% assign C-shaped RH


qkwargs = {
    "CTH < 4 km": {
        "rhmid": 0.5,
        "rhlcl": 0.9,
        "rhtoa": 0.6,
        "Tmin": 280,
        "zlcl": 440,
        "es": es,
        "factor": 0.4,
        "lowlim": 286,
        "highlim": 260,
    },
    "CTH 4-8 km": {
        "rhmid": 0.32,
        "rhlcl": 0.87,
        "rhtoa": 0.4,
        "Tmin": 250,
        "zlcl": 510,
        "es": es,
        "factor": 0.67,
        "lowlim": 287,
        "highlim": 255,
    },
    "CTH > 8 km": {
        "rhmid": 0.42,  # 0.42
        "rhlcl": 0.88,  # 0.9
        "rhtoa": 0.7,  # 0.35
        "Tmin": 250,  # 250
        "zlcl": 543,
        "es": es,
        "factor": 0.52,  # 0.52
        "lowlim": 285,
        "highlim": 255,  # 262
    },
}


qrev = {}
qpseu = {}
ad = "reversible"
rev = adiabat_ds.sel(adiabat=ad)
for cth in ["CTH < 4 km", "CTH 4-8 km", "CTH > 8 km"]:  #
    pseu = adiabat_ds.sel(adiabat="pseudo", cth=cth)
    ds = rev.sel(cth=cth)
    qrev[cth] = ds
    qpseu[cth] = pseu
    for qname, qshape in zip(["c", "e"], [rad.cshape_humidity, rad.wshape_humidity]):
        qrev[cth] = (
            qrev[cth]
            .assign({qname + "q": (("altitude",), qshape(ds, **qkwargs[cth]).values)})
            .assign(cth=cth, adiabat=ad)
        )
        qrev[cth] = qrev[cth].assign(
            {
                qname + "rh": mtf.specific_humidity_to_relative_humidity(
                    q=qrev[cth][qname + "q"], p=qrev[cth].P, T=qrev[cth].T, es=es
                )
            }
        )

        rhfree = qrev[cth].sel(
            altitude=slice(
                qkwargs[cth]["zlcl"], qrev[cth].altitude[qrev[cth].T.argmin()]
            )
        )
        rhfree_pseud = pseu
        rhfree_pseud = rhfree_pseud.sel(
            altitude=slice(
                qkwargs[cth]["zlcl"], rhfree_pseud.altitude[rhfree_pseud.T.argmin()]
            )
        )

        rh_ps = (
            rhfree.swap_dims({"altitude": "T"})
            .interp(T=rhfree_pseud.T.values, kwargs={"fill_value": "extrapolate"})
            .assign(altitude=("T", rhfree_pseud.altitude.values))
            .swap_dims({"T": "altitude"})
            .interp_like(pseu)
            .reset_coords("T")
        )
        qpseu[cth] = qpseu[cth].assign(
            {
                qname + "q": mtf.relative_humidity_to_specific_humidity(
                    rh_ps[qname + "rh"], pseu.P, pseu.T, es=es
                )
                .bfill("altitude")
                .ffill("altitude")
            }
        )
        qpseu[cth] = qpseu[cth].assign(
            {
                qname + "rh": mtf.specific_humidity_to_relative_humidity(
                    qpseu[cth][qname + "q"], qpseu[cth].P, qpseu[cth].T, es=es
                )
            }
        )

    rhpseu = xr.concat([qpseu[cth] for cth in qpseu.keys()], dim="cth")
    adiabat_ds = xr.merge(
        [
            adiabat_ds,
            xr.concat(
                [
                    xr.concat([qpseu[cth] for cth in qpseu.keys()], dim="cth"),
                    xr.concat([qrev[cth] for cth in qrev.keys()], dim="cth"),
                ],
                dim="adiabat",
            ),
        ]
    )

    """
    adiabat_ds = adiabat_ds.assign(
        xr.concat([
        xr.concat([qpseu[cth] for cth in qpseu.keys()], dim="cth"),
        xr.concat([qrev[cth] for cth in qrev.keys()], dim="cth")
        ], dim="adiabat")
    ) 
    """
# %%
cw = 190 / 25.4
sns.set_context("talk", font_scale=0.8)
colors = sns.color_palette("Paired")
fig, ax = plt.subplots()
for idx, cth in enumerate(["CTH < 4 km", "CTH 4-8 km", "CTH > 8 km"]):
    if idx == 0:
        idlabel = "idealized E-shape"
        dotlabel = "idealized C-shape"
    else:
        idlabel = ""
        dotlabel = ""
    for adiabat in ["reversible"]:
        """
        """
        ax.plot(
            adiabat_ds.sel(adiabat=adiabat, cth=cth).erh,
            adiabat_ds.sel(adiabat=adiabat, cth=cth).T,
            label=idlabel,
            color=colors[2 * idx],
        )
        ax.plot(
            adiabat_ds.sel(adiabat=adiabat, cth=cth).crh,
            adiabat_ds.sel(adiabat=adiabat, cth=cth).T,
            color=colors[2 * idx + 1],
            alpha=0.5,
            label=dotlabel,
            linestyle=":",
        )

    ax.plot(
        beachdata[cth].rh.mean("sonde_id"),
        beachdata[cth].ta.mean("sonde_id"),
        label="BEACH " + cth,
        color=colors[2 * idx + 1],
    )


ax.legend()
ax.set_xlim(0, 1)
ax.invert_yaxis()
ax.set_xlabel("RH / 1")
ax.set_ylabel(r"$T$ / K")
sns.despine(offset={"bottom": 10})
fig.savefig(file_path + "mlcloud-rh_idealized_profiles.pdf")


# %%

def calc_radiation(ds, qvar="q"):
    mu0 = angles.get_mu_day(
                    np.datetime64("2024-08-30T00:00:00"), lat=0, lon=-30
                )
    atmosphere = rad.make_atmosphere(
                ds.P.values.reshape(1, ds.P.shape[0]),
                ds.T.values.reshape(1, ds.P.shape[0]),
                ph.specific_humidity2vmr(ds[qvar]).values.reshape(1, ds.P.shape[0]),
                o3=radbeach.O3.interp(altitude=ds.altitude).values,
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

    res = xr.merge(
                [
                    lw_fluxes,
                    xr.concat(sw_fluxes, dim="mu0"),
                    atmosphere,
                ] 
            )

    return ( res
                .assign(
                    {
                    "lw_htgr":xr.apply_ufunc(
                        ph.calc_heating_rate_from_flx,
                        res.lw_flux_up,
                        res.lw_flux_down,
                        res.pres_level,
                        input_core_dims=[["level"], ["level"], ["level"]],
                        output_core_dims=[["level"]],
                        vectorize=True,
                    ),
                    "sw_htgr":xr.apply_ufunc(
                        ph.calc_heating_rate_from_flx,
                        res.sw_flux_up,
                        res.sw_flux_down,
                        res.pres_level,
                        input_core_dims=[["level"], ["level"], ["level"]],
                        output_core_dims=[["level"]],
                        vectorize=True,
                    ),
                    "theta":mtf.theta(
                        res.temp_level,
                        res.pres_level,
                    ),
                    "altitude":("level", ds.altitude.values),
                    qvar:("level", ds[qvar].values),
                    }
                )
                .rename(
                    temp_level="ta",
                    pres_level="p",
                )
                .swap_dims({"level": "altitude"})
            )

cths = []
for cth in ["CTH < 4 km", "CTH 4-8 km", "CTH > 8 km"]:
    ads = []
    for adiabat in ["pseudo", "reversible"]:
        qs = []
        for qvar in ["c", "e"]:
            ds = adiabat_ds.sel(cth = cth, adiabat=adiabat)
            qs.append(calc_radiation(ds, qvar=qvar + "q").assign_coords(
                cth=cth, adiabat=adiabat, rhshape=qvar
            )
            )
        ads.append(
            xr.concat(qs, dim="rhshape")
        )
    cths.append(
        xr.concat(ads, dim="adiabat")
    )

xr.concat(
    cths,
    dim="cth"
).to_netcdf(
    file_path + "idealized_radiation_profiles.nc"
)
    
    
#%%
ds = xr.open_dataset(
    file_path + "idealized_radiation_profiles.nc"
).sel(cth = "CTH 4-8 km")

colors = ["#006C66", "#EF7C00"]

fig, ax = plt.subplots(
    figsize=(cw, 0.5*cw)
)
for ad, ls in zip(["reversible", "pseudo"], ["-", ":"]):
    pltds = ds.sel(adiabat=ad,rhshape="e").sel(column=0)
    ax.plot(60*60 * 24*pltds.lw_htgr, pltds.altitude, label="LW heating rate", c=colors[0], linestyle=ls)
    ax.plot(60*60 * 24*pltds.sw_htgr.mean("mu0"), pltds.altitude, label="SW mean", c=colors[1], linestyle=ls)
    ax.plot(60*60 * 24*pltds.sw_htgr.sel(mu0=12), pltds.altitude, label="SW heatingrate noon", c="red", linestyle=ls)

ax.set_ylim(0, 15000)
ax.set_xlim(-3, 3)
ax.legend()
sns.despine()
