# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from moist_thermodynamics import constants
import sys
from xhistogram.xarray import histogram

sys.path.append("../")
import myutils.open_datasets as open_datasets
from myutils.physics_helper import (
    get_stability,
    density_from_q,
    get_csc_stab,
    get_csc_cooling,
)
import myutils.data_helper as dh

# %%
flux_data_arts2 = open_datasets.open_radiative_fluxes()

cid = "ipfs://bafybeiesyutuduzqwvu4ydn7ktihjljicywxeth6wtgd5zi4ynxzqngx4m"
lvl3 = open_datasets.open_dropsondes(cid)

lvl3 = (
    lvl3.where((lvl3.p_qc == 0) & (lvl3.ta_qc == 0) & (lvl3.rh_qc == 0), drop=True)
    .pipe(dh.interpolate_gaps)
    .pipe(dh.extrapolate_sfc)
)


# %%
stability = (
    get_stability(
        lvl3.swap_dims({"sonde": "sonde_id"}).theta,
        lvl3.swap_dims({"sonde": "sonde_id"}).ta,
    )
    .coarsen(altitude=10)
    .mean()
    .rolling(altitude=10)
    .mean()
)
stability["altitude"] = stability["altitude"] + 5
stability.name = "stability"

rho = (
    density_from_q(
        lvl3.swap_dims({"sonde": "sonde_id"}).p,
        lvl3.swap_dims({"sonde": "sonde_id"}).ta,
        lvl3.swap_dims({"sonde": "sonde_id"}).q,
    )
    .coarsen(altitude=10)
    .mean()
    .rolling(altitude=10)
    .mean()
)

rho["altitude"] = rho["altitude"] + 5
rho.name = "rho"

t = (
    lvl3.set_coords("sonde_id")
    .ta.interpolate_na("altitude", fill_value="extrapolate")
    .coarsen(altitude=10)
    .mean()
    .rolling(altitude=10)
    .mean()
)
t["altitude"] = t["altitude"] + 5
# %%#
# %%
H = flux_data_arts2.cooling_rate * constants.cpv * rho  # J m-3 day-1
# %%
csc_stab = get_csc_stab(rho, stability, H).compute()
csc_stab.name = "csc_stab"
csc_cooling = get_csc_cooling(rho, stability, H).compute()
csc_cooling.name = "csc_cooling"

# %%


histds = flux_data_arts2.sel(altitude=slice(0, 15000))
cr_bins = np.linspace(0, 5, 50)
ta_bins = np.linspace(210, 305, 100)
stab_bins = np.linspace(0, 8, 100)
csc_bins = np.linspace(-5, 5, 100)
# %%
cr_of_ta = histogram(
    histds.ta,
    histds.cooling_rate,
    dim=["altitude"],
    bins=[ta_bins, cr_bins],
).compute()
# %%
stab_of_ta = histogram(
    t.swap_dims({"sonde": "sonde_id"}),
    stability,
    dim=["altitude"],
    bins=[ta_bins, stab_bins],
).compute()
# %%
tnew = (
    t.isel(altitude=slice(0, -1))
    .where(t.sonde_id.isin(csc_stab.sonde_id), drop=True)
    .swap_dims({"sonde": "sonde_id"})
)
# %%
csc_stab_of_ta = histogram(
    tnew,
    csc_stab,
    dim=["altitude"],
    bins=[ta_bins, csc_bins],
).compute()
# %%
csc_cooling_of_ta = histogram(
    tnew,
    csc_cooling,
    dim=["altitude"],
    bins=[ta_bins, csc_bins],
).compute()

# %%
fig, ax = plt.subplots()
name = "csc_stab"
for name in ["csc_cooling", "csc_stab"]:
    pltval = eval(f"{name}_of_ta").mean("sonde_id").rolling(ta_bin=5).mean()
    pltval = (pltval * pltval[f"{name}_bin"]).sum(f"{name}_bin") / pltval.sum(
        f"{name}_bin"
    )

    pltval.rolling(ta_bin=5).mean().plot(
        y="ta_bin",
    )
pltval = (
    (csc_cooling_of_ta.rename({"csc_cooling_bin": "csc_stab_bin"}) + csc_stab_of_ta)
    .mean("sonde_id")
    .rolling(ta_bin=5)
    .mean()
)
pltval = (pltval * pltval[f"csc_stab_bin"]).sum(f"csc_stab_bin") / pltval.sum(
    f"csc_stab_bin"
)

pltval.rolling(ta_bin=5).mean().plot(
    y="ta_bin",
    color="k",
)
ax.invert_yaxis()
ax.axvline(0, color="grey", linestyle="--")
ax.axhline(273.15, color="grey", linestyle="--")
