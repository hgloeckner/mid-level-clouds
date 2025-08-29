# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import moist_thermodynamics.functions as mt
from moist_thermodynamics import saturation_vapor_pressures as svp
import xarray as xr
from moist_thermodynamics import constants

from pydropsonde.circles import Circle
import sys

sys.path.append("../")
import myutils.open_datasets as opends  # noqa
from myutils.physics_helper import get_n2, get_stability

es = mt.make_es_mxd(es_liq=svp.liq_analytic, es_ice=svp.ice_analytic)
# %%
cid = opends.get_cid()
rs = opends.open_radiosondes(f"{cid}/Radiosondes/RAPSODI_RS_ORCESTRA_level2.zarr")

ds = opends.open_dropsondes(f"{cid}/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr")

gate = opends.open_gate(opends.get_gate_cid())

unique, keep = np.unique(gate.sonde_id.values, return_index=True)
gate = gate.isel(sonde_id=keep)
datasets = {
    "radiosonde": (
        Circle(
            rs,
            alt_dim="altitude",
            sonde_dim="sonde_id",
            clon=None,
            clat=None,
            crad=None,
            flight_id=None,
            platform_id=None,
            segment_id=None,
        )
        .drop_vars()
        .interpolate_na_sondes()
        .extrapolate_na_sondes()
        .circle_ds.where(rs.ascent_flag == 0, drop=True)
    ),
    "dropsonde": (
        Circle(
            ds,
            alt_dim="altitude",
            sonde_dim="sonde_id",
            clon=None,
            clat=None,
            crad=None,
            flight_id=None,
            platform_id=None,
            segment_id=None,
        )
        .drop_vars()
        .interpolate_na_sondes()
        .extrapolate_na_sondes()
        .circle_ds
    ),
    "gate": (
        Circle(
            gate,
            alt_dim="altitude",
            sonde_dim="sonde_id",
            clon=None,
            clat=None,
            crad=None,
            flight_id=None,
            platform_id=None,
            segment_id=None,
        )
        .drop_vars()
        .interpolate_na_sondes()
        .extrapolate_na_sondes()
        .circle_ds
    ),
}
# %%
for dataname, data in datasets.items():
    datasets[dataname] = data.assign(
        ta=data.ta.interpolate_na("altitude", fill_value="extrapolate", max_gap=800),
    )


# %%

es_liq = svp.liq_analytic
es_ice = svp.ice_analytic
es_mixed = mt.make_es_mxd(es_liq=svp.liq_analytic, es_ice=svp.ice_analytic)
Rd = constants.Rd
Rv = constants.Rv
cpd = constants.cpd
P0 = constants.P0


def get_rh(ta, theta, q, es=es):
    ds_p = (ta / theta) ** (cpd / Rd) * P0
    R = Rd + (Rv - Rd) * q
    pv = q * Rv / R * ds_p
    return pv / es(ta)


# %%
sns.set_palette("Paired")
fig, ax = plt.subplots(1, 1, figsize=(4, 5), sharey=True)
# plot radiosondes
for dataname, data in datasets.items():
    rh = get_rh(data.ta, data.theta, data.q)
    rh.mean("sonde_id").plot(
        y="altitude", xlim=(0, 1), ylim=(0, 23000), label=f"ice-liq {dataname}"
    )
    rh = get_rh(data.ta, data.theta, data.q, es=svp.liq_analytic)
    rh.mean("sonde_id").plot(
        y="altitude", xlim=(0, 1), ylim=(0, 23000), label=f"ice-liq {dataname}"
    )

ax.set_ylabel("altitude / m")
ax.set_xlabel("RH")
sns.despine(offset=10)
plt.legend()
# %%

# rs = rs.where((rs.rh < 1) & (rs.rh >0))
for dataname, data in datasets.items():
    datasets[dataname] = data.assign(
        n2=get_n2(
            th=data.theta,
            qv=data.q,
        ),
        stability=get_stability(
            theta=data.theta,
            T=data.ta,
        ),
    )
# %%
for dataname, data in datasets.items():
    datasets[dataname] = data.assign(
        theta_e_bolton_mixed=mt.theta_e_bolton(
            T=data.ta,
            P=data.p,
            qt=data.q,
            es=es_mixed,
        ),
        theta_e_bolton_liq=mt.theta_e_bolton(
            T=data.ta,
            P=data.p,
            qt=data.q,
            es=svp.liq_analytic,
        ),
        plcl_bolton=mt.plcl_bolton(
            T=data.ta.sel(altitude=0),
            P=data.p.sel(altitude=0),
            qt=data.q.sel(altitude=0),
        ),
    )


# %%
def get_var_at_lcl(da, p, plcl):
    def get_lcl_var(da, p, plcl):
        lcl_var = da.where(np.abs(p - plcl) == np.min(np.abs(p - plcl)), drop=True)

        try:
            return lcl_var.squeeze(dim="altitude", drop=True)
        except IndexError:
            return xr.DataArray(
                np.nan,
                dims=[],
                coords={
                    coord: da[coord].values
                    for coord in da.coords
                    if coord != "altitude"
                },
                name=da.name,
            )

    result = []
    for sonde_id in da.sonde_id.values:
        result.append(
            get_lcl_var(
                da.sel(sonde_id=sonde_id),
                p.sel(sonde_id=sonde_id),
                plcl.sel(sonde_id=sonde_id),
            )
        )
    return xr.concat(result, dim="sonde_id")


# %%


def get_t_of_theta_e_xr(f, theta_e_lcl, P, qlcl, es):
    """Returns the Brunt-Vaisala frequeny for unsaturated air.

    It assumes that the input are type xarray with their first coordinate being
    altitude in meters, and that the air is no where saturated

    Args:
        th: potential temperature
        qv: specific humidity
    """

    def invert_for_temperature(P, f, theta_e, qt, es):
        """Inverts the function f for temperature."""
        try:
            return mt.invert_for_temperature(
                f=f,
                f_val=theta_e,
                P=P,
                qt=qt,
                es=es,
            )
        except RuntimeError:
            return np.nan

    return xr.apply_ufunc(
        invert_for_temperature,
        P,
        f,
        theta_e_lcl,
        qlcl,
        es,
        input_core_dims=[
            [],
            [],
            [],
            [],
            [],
        ],
        output_core_dims=[[]],
        vectorize=True,
        # dask="parallelized",
    )


# %%
for dataname, data in datasets.items():
    try:
        stability = xr.open_dataset(
            f"/work/mh0066/m301046/{dataname}_idealized_stability.nc",
        )
        datasets[dataname] = xr.merge([data, stability])
    except FileNotFoundError:
        for dataset_name, dataset in datasets.items():
            datasets[dataset_name] = dataset.assign(
                theta_e_lcl=get_var_at_lcl(
                    da=dataset.theta_e_bolton_mixed,
                    p=dataset.p,
                    plcl=dataset.plcl_bolton,
                ),
                q_lcl=get_var_at_lcl(
                    da=dataset.q,
                    p=dataset.p,
                    plcl=dataset.plcl_bolton,
                ),
            )

        es = es_mixed
        for dataname, data in datasets.items():
            print(dataname)
            datasets[dataname] = data.assign(
                idealized_stability=get_n2(
                    th=mt.theta(
                        T=get_t_of_theta_e_xr(
                            f=mt.theta_e_bolton,
                            theta_e_lcl=data.theta_e_lcl.compute(),
                            P=data.p,
                            qlcl=data.q_lcl,
                            es=es,
                        ).compute(),
                        P=data.p,
                        qv=data.q,
                    ),
                    qv=data.q,
                    altdim="altitude",
                )
            )
            datasets[dataname][
                ["idealized_stability", "q_lcl", "theta_e_lcl", "plcl_bolton"]
            ].to_netcdf(
                f"/work/mh0066/m301046/{dataname}_idealized_stability.nc",
            )
# %%

# %%
var = "n2"
sns.set_palette("colorblind")
fig, ax = plt.subplots(figsize=(6, 5), sharey=True)

for dataname, data in datasets.items():
    # data.stability.mean("sonde_id").plot(
    #     label=dataname,
    #     y="altitude",
    # )
    ax.plot(
        data[var]
        .mean("sonde_id")
        .sel(altitude=slice(0, 14000))
        .rolling(altitude=50)
        .mean()
        .values,
        data.ta.mean("sonde_id")
        .sel(altitude=slice(0, 14000))
        .rolling(altitude=50)
        .mean()
        .values,
        label=dataname,
    )
ax.plot(
    data.idealized_stability.mean("sonde_id")
    .rolling(altitude=15)
    .mean()
    .sel(altitude=slice(0, 14000)),
    data.ta.mean("sonde_id").rolling(altitude=15).mean().sel(altitude=slice(0, 14000)),
    label="pseudo",
    color="black",
)

ax.legend()
ax.invert_yaxis()
# ax.set_ylim(295, 220)
ax.set_xlim(0.005, 0.017)
ax.axhline(273.15, color="k", linestyle="-", lw=0.5)
sns.despine(offset={"bottom": 10})
# %%

# %%
