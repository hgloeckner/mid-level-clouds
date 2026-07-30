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
import myutils.moist_adiabats as ma
from radiation_for_sondes.rrtmg import angles
import radiation_for_sondes.rrtmg.rad_helper as rad

es = mtf.make_es_mxd(svp.liq_wagner_pruss, svp.ice_wagner_etal)

levante = False

if levante:
    filepath = "/scratch/m/m301046/"
    cth_path = "/work/mh0066/m301046/ml_clouds/"

else:
    file_path = "/Users/helene/Documents/code/mid_level_clouds/plots/"
    cth_path = file_path + "sondes_for_radiation.nc"


radds = xr.open_dataset(file_path + "idealized_radiation_profiles.nc")
realds = xr.open_dataset(file_path + "real_radiation_profiles.nc")


# %%
def calc_cs_convergence(ds, ta_var="ta", qvar="cq"):
    res = {}
    res["stab"] = ph.get_stability(ds.theta, ds[ta_var])  # .rolling(altitude=10).mean()
    res["rho"] = ph.density_from_q(
        ds.p, ds[ta_var], ds[qvar]
    )  # .rolling(altitude=10).mean()
    htgr = ds.lw_htgr + ds.sw_htgr.mean("mu0")  # .rolling(altitude=10).mean()
    res["csc_stab"] = ph.get_csc_stab(res["rho"], res["stab"], htgr)
    res["csc_cool"] = ph.get_csc_cooling(res["rho"], res["stab"], htgr)
    res["mass_flux"] = ph.mass_flux(res["rho"], res["stab"], htgr)
    htgr = ds.lw_htgr  # .rolling(altitude=10).mean()
    res["csc_stab_lw"] = ph.get_csc_stab(res["rho"], res["stab"], htgr)
    res["csc_cool_lw"] = ph.get_csc_cooling(res["rho"], res["stab"], htgr)
    res["lw_mass_flux"] = ph.mass_flux(res["rho"], res["stab"], htgr)
    htgr = ds.lw_htgr + ds.sw_htgr.sel(mu0=12)  # .rolling(altitude=10).mean()
    res["csc_stab_swmax"] = ph.get_csc_stab(res["rho"], res["stab"], htgr)
    res["csc_cool_swmax"] = ph.get_csc_cooling(res["rho"], res["stab"], htgr)
    return res


def get_heating_from_mass_flux(mass_flux, stability, rho):
    return mass_flux * stability / rho


csc = dict(
    c=calc_cs_convergence(radds, qvar="cq"),
    e=calc_cs_convergence(radds, qvar="eq"),
    creal=calc_cs_convergence(realds, qvar="cq"),
    ereal=calc_cs_convergence(realds, qvar="eq"),
)


# %%
cth = "CTH < 4 km"
ad = "reversible"
colors = ["#006C66", "k", "#EF7C00"]
cw = 190 / 25.4
sns.set_context("talk", font_scale=0.8)

fig, axs = plt.subplots(
    nrows=2, ncols=3, figsize=(cw, 0.8 * cw), sharex="col", sharey=True
)

axes = axs[0]
select = dict(cth=cth, column=0, altitude=slice(0, 12000))

for axes, ds, suff, seldict, roll in zip(
    [axs[0], axs[1]], [radds, realds], ["", "real"], [{"adiabat": ad}, {}], [1, 10]
):
    for ll, ls, qvar in zip(["C shape", "E shape"], [":", "-"], ["c", "e"]):
        select["rhshape"] = qvar
        pltds = ds.sel(select).sel(seldict).rolling(altitude=roll).mean()
        axes[0].plot(
            pltds.lw_htgr * 60 * 60 * 24,
            pltds.ta,
            color=colors[0],
            linestyle=ls,
        )
        axes[0].plot(
            (pltds.lw_htgr + pltds.sw_htgr.mean("mu0")) * 60 * 60 * 24,
            pltds.ta,
            color="k",
            linestyle=ls,
            label=ll,
        )
        axes[0].plot(
            (pltds.lw_htgr + pltds.sw_htgr.sel(mu0=12)) * 60 * 60 * 24,
            pltds.ta,
            color=colors[2],
            linestyle=ls,
        )

        pltcsc = csc[qvar + suff]
        for idx, (end, color, label) in enumerate(
            zip(["_lw", "", "_swmax"], colors, ["LW", "LW + mean(SW)", "LW + max(SW)"])
        ):
            if qvar == "e":
                pltlabel = label
            else:
                pltlabel = ""
            pltrho = (
                pltcsc["rho"]
                .sel(select)
                .sel(seldict)
                .rolling(altitude=roll)
                .mean()
                .values
            )
            axes[1].plot(
                (
                    pltcsc[f"csc_cool{end}"]
                    .sel(select)
                    .sel(seldict)
                    .rolling(altitude=roll)
                    .mean()
                    .values
                )
                / pltrho
                * 60
                * 60
                * 24,
                pltds.ta,
                color=color,
                linestyle=ls,
                label=pltlabel,
            )
            axes[2].plot(
                (
                    pltcsc[f"csc_cool{end}"]
                    .sel(select)
                    .sel(seldict)
                    .rolling(altitude=roll)
                    .mean()
                    + pltcsc[f"csc_stab{end}"]
                    .sel(select)
                    .sel(seldict)
                    .rolling(altitude=roll)
                    .mean()
                ).values
                / pltrho
                * 60
                * 60
                * 24,
                pltds.ta,
                color=color,
                linestyle=ls,
            )

axes[0].set_ylim(250, 290)
axes[0].set_xlim(-3.5, 1)
axes[0].set_xlabel("$\\mathcal{{H}}$ / K day$^{-1}$")
axes[1].set_xlim(-0.4, 0.3)
axes[1].set_xlabel(r"CSC$_{\mathcal{H}}$ / day$^{-1}$")
axes[2].set_xlim(-0.4, 0.3)
axes[2].set_xlabel(r"CSC$_{\mathcal{H}}$ + CSC$_{\sigma}$" + "\n / day$^{-1}$")
for ax in axs[:, 0]:
    ax.set_ylabel("$T$ / K")
axs[0, 0].invert_yaxis()
for ax in axs.flatten():
    ax.axhline(273.15, color="gray", alpha=0.3)
    ax.axvline(0, color="gray", alpha=0.3)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=1)
sns.despine(offset=10)
fig.tight_layout()
axs[0, 0].legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.2, 1), framealpha=1)
axs[1, 1].legend(fontsize=10, loc="upper left", bbox_to_anchor=(-0.6, 1), framealpha=1)
# fig.savefig(file_path + "mlcloud-csc.pdf")
# %%


# %%


def get_heating_from_mass_flux(mass_flux, stability, rho):
    return mass_flux * stability / rho


def get_vertical_mean(da, mean_dim):
    dz = da.swap_dims({"altitude": mean_dim})[mean_dim].differentiate(mean_dim)
    return da.swap_dims({"altitude": mean_dim}).weighted(dz).mean(mean_dim)


cth = "CTH < 4 km"
ad = "reversible"
colors = ["#006C66", "k", "#EF7C00"]
fig, axs = plt.subplots(
    ncols=3, nrows=2, figsize=(cw, 0.6 * cw), sharex="col", sharey=True
)

select = dict(cth=cth, column=0, altitude=slice(0, 15000))

for axes, ds, suff, seldict in zip(
    [axs[0], axs[1]], [radds, realds], ["", "real"], [{"adiabat": ad}, {}]
):
    for ll, ls, qvar in zip(["C shape", "E shape"], [":", "-"], ["c", "e"]):
        select["rhshape"] = qvar
        pltds = ds.sel(select).sel(seldict)
        pltcsc = csc[qvar + suff]
        axes[0].plot(
            pltcsc["mass_flux"].sel(select).sel(seldict) * 60 * 60,
            pltds.ta,
            linestyle=ls,
            color=colors[2],
        )
        axes[1].plot(
            (pltds.lw_htgr + pltds.sw_htgr.mean("mu0")) * 60 * 60 * 24,
            pltds.ta,
            linestyle=ls,
            color=colors[2],
        )
        mean_mass = get_vertical_mean(
            pltcsc["mass_flux"]
            .sel(select)
            .sel(seldict)
            .assign_coords(ta=pltds.ta)
            .sel(altitude=slice(1000, None)),
            mean_dim="ta",
        ).broadcast_like(pltds.ta)
        new_htgr = get_heating_from_mass_flux(
            mean_mass,
            pltcsc["stab"].sel(select).sel(seldict),
            pltcsc["rho"].sel(select).sel(seldict),
        )
        axes[0].plot(mean_mass * 60 * 60, pltds.ta, linestyle=ls, color=colors[0])
        axes[1].plot(new_htgr * 60 * 60 * 24, pltds.ta, linestyle=ls, color=colors[0])
        axes[2].plot(
            ((pltds.lw_htgr + pltds.sw_htgr.mean("mu0")) - new_htgr) * 60 * 60 * 24,
            pltds.ta,
            linestyle=ls,
            color="k",
        )
        mean_mass_lw = get_vertical_mean(
            pltcsc["lw_mass_flux"]
            .sel(select)
            .sel(seldict)
            .assign_coords(ta=pltds.ta)
            .sel(altitude=slice(1000, None)),
            mean_dim="ta",
        ).broadcast_like(pltds.ta)
        new_htgr = get_heating_from_mass_flux(
            mean_mass_lw,
            pltcsc["stab"].sel(select).sel(seldict),
            pltcsc["rho"].sel(select).sel(seldict),
        )
        axes[2].plot(
            (pltds.lw_htgr - new_htgr) * 60 * 60 * 24,
            pltds.ta,
            linestyle=ls,
            color="grey",
        )


axs[0, 0].set_xlim(-15, -2)

axs[1, 1].set_xlabel("$\mathcal{H}$ / K day$^{-1}$")
axs[1, 0].set_xlabel("$M$ / kg m$^{-2}$ hr$^{-1}$")
axs[1, 2].set_xlabel("$\Delta\mathcal{H} $/ K day$^{-1}$ ")

axs[0, 1].set_xlim(-2.5, -0.5)
axs[0, 0].set_ylim(250, 290)
axs[0, 0].invert_yaxis()
axs[0, 2].set_xlim(-1, 1)
for ax in axs[:, 2]:
    ax.axvline(0, color="grey", alpha=0.3)
for ax in axs.flatten():
    ax.axhline(273.15, color="grey", alpha=0.3)
for ax in axs[:, 0]:
    ax.set_ylabel("T / K")

sns.despine()
sns.despine(offset=10)
fig.savefig(file_path + "mlcloud-mflux.pdf")
# %%


# %%
cth = "CTH < 4 km"
ad = "reversible"
bothshape = True
colors = ["#006C66", "k", "#EF7C00"]
cw = 190 / 25.4
sns.set_context("talk", font_scale=0.8)
# sns.set_context("paper")
kwargs = {"linewidth": 2}

file_end = ""
select = dict(cth=cth, column=0, altitude=slice(0, 12000))
seldict = {"adiabat": ad}
for i in range(4):
    fig, axes = plt.subplots(ncols=3, figsize=(cw, 0.5 * cw), sharey=True)
    for ll, ls, qvar in zip(["C shape", "E shape"], [":", "-"], ["c", "e"]):
        if not bothshape:
            if qvar == "e":
                y = np.full_like(pltds.ta, np.nan)
                ll = ""
            else:
                y = pltds.ta
            file_end = "_c"

        else:
            y = pltds.ta
        select["rhshape"] = qvar
        pltds = radds.sel(select).sel(seldict)
        if i > 0:
            axes[0].plot(
                pltds.lw_htgr * 60 * 60 * 24,
                y,
                color=colors[0],
                linestyle=ls,
                **kwargs,
            )
        if i > 1:
            axes[0].plot(
                (pltds.lw_htgr + pltds.sw_htgr.mean("mu0")) * 60 * 60 * 24,
                y,
                color="k",
                linestyle=ls,
                label=ll,
                **kwargs,
            )
        if i > 2:
            axes[0].plot(
                (pltds.lw_htgr + pltds.sw_htgr.sel(mu0=12)) * 60 * 60 * 24,
                y,
                color=colors[2],
                linestyle=ls,
                **kwargs,
            )

        pltcsc = csc[qvar]
        for idx, (end, color, label) in enumerate(
            zip(["_lw", "", "_swmax"], colors, ["Night", "24 hr mean", "Noon"])
        ):
            if end == "" or i == 1:
                pltlabel = ll
            else:
                pltlabel = ""
            if qvar == "e":
                clabel = label
            else:
                clabel = ""
            pltrho = pltcsc["rho"].sel(select).sel(seldict)
            if idx < i:
                axes[1].plot(
                    (pltcsc[f"csc_cool{end}"].sel(select).sel(seldict).values)
                    / pltrho
                    * 60
                    * 60
                    * 24,
                    y,
                    color=color,
                    linestyle=ls,
                    label=clabel,
                    **kwargs,
                )
                axes[2].plot(
                    (
                        pltcsc[f"csc_cool{end}"].sel(select).sel(seldict)
                        + pltcsc[f"csc_stab{end}"].sel(select).sel(seldict)
                    ).values
                    / pltrho
                    * 60
                    * 60
                    * 24,
                    y,
                    color=color,
                    linestyle=ls,
                    label=pltlabel,
                    **kwargs,
                )

    axes[0].set_ylim(250, 290)
    axes[0].set_xlim(-3.5, 1)
    axes[0].set_xlabel("$\\mathcal{{H}}$ / K day$^{-1}$")
    axes[1].set_xlim(-0.4, 0.3)
    axes[1].set_xlabel(r"CSC$_{\mathcal{H}}$ / day$^{-1}$")
    axes[2].set_xlim(-0.4, 0.3)
    axes[2].set_xlabel(
        r"CSC$_{\mathcal{H}}$ + CSC$_{\sigma}$" + "\n / day$^{-1}$"
    )  # \n
    for ax in axs[:, 0]:
        ax.set_ylabel("$T$ / K")
    axes[0].invert_yaxis()
    for ax in axes:
        ax.axhline(273.15, color="k", alpha=0.9, linewidth=0.5, zorder=1)
        ax.axvline(0, color="k", alpha=0.9, linewidth=0.5, zorder=1)
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_linewidth(1)
        ax.xaxis.set_tick_params(width=1)
        ax.yaxis.set_tick_params(width=1)
    axes[0].set_ylabel("$T$ / K")
    sns.despine(offset=10)
    fig.tight_layout()

    axes[1].legend(
        fontsize=10, loc="upper left", bbox_to_anchor=(-0.4, 1), framealpha=1
    )
    axes[2].legend(
        fontsize=10, loc="upper left", bbox_to_anchor=(-0.3, 1), framealpha=1
    )
    fig.savefig(
        file_path + f"csc_{i}{file_end}.pdf", bbox_inches="tight", transparent=True
    )
