# %%
#
import xarray as xr
import numpy as np
import sys

sys.path.append("mlclouds")
import myutils.physics_helper as ph
import moist_thermodynamics.functions as mtf
import moist_thermodynamics.saturation_vapor_pressures as svp
import matplotlib.pyplot as plt
import seaborn as sns

# %%


es = mtf.make_es_mxd(svp.liq_wagner_pruss, svp.ice_wagner_etal)
# %%

Px = 101000
P = np.arange(Px, 4000.0, -500)
Tsfc = [290, 288]
rhsfc = 0.9
qsfc = [mtf.relative_humidity_to_specific_humidity(rhsfc, Px, T, es=es) for T in Tsfc]

plcl = [mtf.plcl_bolton(T=T, P=Px, qt=q) for T, q in zip(Tsfc, qsfc)]
zlcl = [mtf.zlcl(p, T=T, P=Px, qt=q, z=0) for p, T, q in zip(plcl, Tsfc, qsfc)]

adiabats = [
    ph.make_sounding_from_adiabat(P, T, q).rename({"Trho": "ta", "P": "p"})
    for T, q in zip(Tsfc, qsfc)
]
# %%
for adiabat in adiabats:
    plt.plot(adiabat.ta, adiabat.p)
# %%
qkwargs = {
    "rhmid": 0.5,
    "rhlcl": 0.9,
    "rhtoa": 0.9,
    "Tmin": 260,
    "zlcl": zlcl[0],
    "es": es,
    "factor": 0.4,
    "lowlim": 286,
    "highlim": 260,
}

qs = [
    ph.cshape_humidity(
        adiabat,
        rhmid=0.5,
        rhlcl=0.9,
        rhtoa=0.9,
        Tmin=260,
        zlcl=z,
        es=es,
        factor=0.4,
        lowlim=286,
        highlim=260,
    )
    for z, adiabat in zip(zlcl, adiabats)
]
rhs = [
    mtf.specific_humidity_to_relative_humidity(q, adiabat.p, adiabat.ta, es=es)
    for q, adiabat in zip(qs, adiabats)
]

adiabats[0] = adiabats[0].assign(
    rh=rhs[0],
    q=qs[0],
)


def blend_rhs(temp2, rh1, rh2, tlcl2):
    return np.where(
        temp2 < 270,
        rh1,
        np.where(temp2 > tlcl2, rh2, rh1 + (rh2 - rh1) * (temp2 - 270) / (tlcl2 - 270)),
    )


adiabats[1] = adiabats[1].assign(
    rh=(
        "altitude",
        blend_rhs(
            adiabats[1].ta.values,
            adiabats[0]
            .swap_dims({"altitude": "ta"})
            .interp(ta=adiabats[1].ta.values)
            .rh.values,
            rhs[1].values,
            adiabats[1].ta.sel(altitude=zlcl[1], method="nearest").values,
        ),
    )
)

# %%

kwargs = {"linewidth": 3}
colors = ["#005555", "#EF7C00"]
sns.set_context("talk")
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
for adiabat, T, altend, color in zip(
    adiabats[::-1], Tsfc[::-1], [11200, 12000], colors
):
    axes[1].plot(
        adiabat.rh,
        adiabat.ta.where(adiabat.altitude < altend),
        color=color,
        label=f"{T} K",
        **kwargs,
    )
    axes[0].plot(
        adiabat.rh.where(adiabat.altitude < altend),
        adiabat.p / 100,
        color=color,
        **kwargs,
    )
    # break

axes[0].invert_yaxis()
axes[1].invert_yaxis()
axes[1].legend(title="$T$ surface", loc="upper left")
for ax in axes:
    ax.set_xlim(0.3, 0.95)
    ax.set_xlabel("relative humidity / 1")
axes[0].set_ylabel("pressure / hPa")
axes[1].set_ylabel("temperature / K")
axes[0].set_ylim(None, 180)
axes[1].set_ylim(295, 197)
axes[1].set_yticks([290, 273, 250, 201])
axes[0].set_yticks([1010, 735, 485, 210])
sns.despine(offset=10)
fig.tight_layout()
fig.savefig(
    "/Users/helene/Documents/Seafile/presentation/defense/figures/idealized_rh.pdf",
    bbox_inches="tight",
)

# %%
for adiabat in adiabats:
    plt.plot(adiabat.ta, adiabat.altitude)
plt.ylim(0, 15000)

# %%

import xarray as xr

ds = xr.open_dataset(
    "ipfs://bafybeiczbv7mycr2jois6t4dq3zwiltycomwo5xxvjqcjz2ot3newzar6q", engine="zarr"
)
ds.rh.mean("sonde").plot(y="altitude")
