# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.constants as mtc
import moist_thermodynamics.saturation_vapor_pressures as svp
import myutils.moist_adiabats as ma
import radiation_for_sondes.rrtmg.rad_helper as rad

es = mtf.make_es_mxd(svp.liq_wagner_pruss, svp.ice_wagner_etal)

# %%
Psfc = 101000.0
P = np.arange(Psfc, 4000.0, -500)
Ts = [288, 290]
cshape = {}
adiabat = {}
zlcl = {}
for name, sfcT in zip(["cold", "warm"], Ts):
    qsfc = mtf.relative_humidity_to_specific_humidity(0.8, Psfc, sfcT, es=es)
    print(qsfc)

    adiabat[name] = ma.make_sounding_from_adiabat(
        ma_fct=ma.pseudo_adiabat,
        P=P,
        Tsfc=sfcT,
        qsfc=qsfc,
        Tmin=195,
    )
    plcl = mtf.plcl(sfcT, Psfc, qsfc, es=es)
    zlcl[name] = mtf.zlcl(plcl, sfcT, Psfc, qsfc, 0)
refname = "cold"
name = "warm"
cshape = rad.cshape_humidity(
    adiabat[refname],
    zlcl=zlcl[refname],
    rhmid=0.4,
    rhlcl=0.9,
    rhtoa=0.8,
    Tmin=240,
    es=es,
)

range = slice(
    zlcl[refname][0], adiabat[refname].altitude[adiabat[refname].T.argmin()].values
)

freerh = mtf.specific_humidity_to_relative_humidity(
    q=cshape.sel(altitude=range),
    p=adiabat[refname].sel(altitude=range).P,
    T=adiabat[refname].sel(altitude=range).T,
    es=es,
).assign_coords(T=adiabat[refname].sel(altitude=range).T)

newrange = slice(zlcl[name][0], adiabat[name].altitude[adiabat[name].T.argmin()].values)
newrh = (
    freerh.swap_dims({"altitude": "T"})
    .interp(
        T=adiabat[name].sel(altitude=newrange).T.values,
        kwargs={"fill_value": "extrapolate"},
    )
    .assign_coords(altitude=("T", adiabat[name].sel(altitude=newrange).altitude.values))
    .swap_dims({"T": "altitude"})
    .interp_like(adiabat[name])
    .reset_coords("T", drop=True)
)

newcshape = mtf.specific_humidity_to_relative_humidity(
    q=mtf.relative_humidity_to_specific_humidity(
        RH=newrh,
        p=P,
        T=adiabat[name].T,
        es=es,
    )
    .bfill("altitude")
    .ffill("altitude"),
    p=P,
    T=adiabat[name].T,
    es=es,
)
# %%

colors = ["#006C66", "#EF7C00"]
cw = 190 / 25.4
sns.set_context("talk", font_scale=0.8)
fig, axes = plt.subplots(ncols=2, figsize=(cw, 0.6 * cw))
axes[0].plot(
    mtf.specific_humidity_to_relative_humidity(cshape, P, adiabat[refname].T, es=es),
    P / 100,
    color=colors[0],
    label=r"$T_{\text{s}} = $" + str(Ts[0]),
)
axes[0].plot(
    newcshape, P / 100, color=colors[1], label=r"$T_{\text{s}} = $" + str(Ts[1])
)
axes[1].plot(
    mtf.specific_humidity_to_relative_humidity(cshape, P, adiabat[refname].T, es=es),
    adiabat[refname].T,
    color=colors[0],
)
axes[1].plot(newcshape, adiabat[name].T, color=colors[1])
axes[0].invert_yaxis()
axes[1].invert_yaxis()
axes[0].legend()
for ax in axes:
    ax.set_xlabel("RH / 1")
    ax.set_xlim(0, 1)
axes[0].set_ylabel(r"$p$ / hPa")
axes[1].set_ylabel(r"$T$ / K")

sns.despine(offset={"bottom": 10})
fig.savefig("../../plots/idealized_rh.pdf")
# %%
