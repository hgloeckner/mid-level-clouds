#%%
import time

import numpy as np
import xarray as xr
import pyarts
import myutils.physics_helper as ph
import calc_flux as cf
import FluxSimulator as fsm

rrtmg_input = xr.open_dataset(
    "/scratch/m/m301046/idealized_radiation_profiles.nc"
)

idealized_sonde_input = xr.open_dataset(
    "/work/mh0066/m301046/ml_clouds/sondes_for_radiation.nc"
)
rrtmg_input = (rrtmg_input.assign(
    q = xr.concat(
        [
            rrtmg_input["cq"].sel(rhshape="c"),
            rrtmg_input["eq"].sel(rhshape="e"),
        ],
        dim="rhshape",
    ),
    o3 = idealized_sonde_input["O3"].interp(altitude=rrtmg_input.altitude),
)
    .drop_vars(["cq", "eq"])
    .sel(cth="CTH < 4 km", adiabat="pseudo")
    .squeeze()
    .drop_vars(["column", "adiabat", "cth"])
    .drop_dims(["mu0"])
)
min_wavenumber = 1
max_wavenumber = 3000
wave_bands = 10000


# %%
#%%
def init_calc(ds):
    atms_grd = pyarts.arts.ArrayOfGriddedField4()
    for i in range(ds.rhshape.size):
        profile = ds.isel(rhshape=i)
        print(ph.specific_humidity2vmr(ds["q"]).values.shape)
        profile_grd = fsm.generate_gridded_field_from_profiles(
            profile["p"].values,
            profile["ta"].values,
            gases={
                "H2O": ph.specific_humidity2vmr(profile["q"]),
                "CO2": profile["co2"],
                "O3": profile["o3"],
                "N2": profile["n2"],
                "O2": profile["o2"],
                "CH4": profile["ch4"],
            },
            z_field=profile["altitude"].values,
        )
        atms_grd.append(profile_grd)

    # Setup Flux Simulator
    f_grid = np.linspace(
        min_wavenumber, max_wavenumber, wave_bands
    )  # frequency grid in cm^-1
    f_grid_freq = pyarts.arts.convert.kaycm2freq(f_grid)  # converted to Hz

    species = [
        "H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
        "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
        "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
        "CO2, CO2-CKDMT252",
        "O3",
        "O3-XFIT",
    ]

    flux_simulator = fsm.FluxSimulator(f"100m_1000m_{wave_bands}_wavebands")
    flux_simulator.ws.f_grid = f_grid_freq
    flux_simulator.set_species(species)
    flux_simulator.get_lookuptableBatch(atms_grd)
    return f_grid, flux_simulator, atms_grd
def create_ds(ds, f_grid, helper_dims):
    shape_flux = tuple([ds.sizes[dim] for dim in helper_dims]) + (ds.sizes["altitude"], len(f_grid))
    shape_integrated = tuple([ds.sizes[dim] for dim in helper_dims]) + (ds.sizes["altitude"],)

    fluxes = xr.Dataset(
        {
            "lw_flux_up_spectral": (
                tuple(helper_dims) + ("altitude", "f_grid"),
                np.full(shape_flux, np.nan),
            ),
            "lw_flux_down_spectral": (
                tuple(helper_dims) + ("altitude", "f_grid"),
                np.full(shape_flux, np.nan),
            ),
            "lw_flux_up": (tuple(helper_dims) + ("altitude",), np.full(shape_integrated, np.nan)),
            "lw_flux_down": (tuple(helper_dims) + ("altitude",), np.full(shape_integrated, np.nan)),
            "heating_rate": (tuple(helper_dims) + ("altitude",), np.full(shape_integrated, np.nan)),
        },
        coords={
            **{dim: (dim, ds[dim].values) for dim in helper_dims},
            "altitude": ("altitude", ds.altitude.values),
            "f_grid": ("f_grid", f_grid),
        },
    )
    return xr.merge([fluxes, ds], compat="override")


f_grid, flux_simulator, atms_grd = init_calc(rrtmg_input)
lw_fluxes = create_ds(rrtmg_input, f_grid, helper_dims=["rhshape"])
# Run simulation and store results
start_time = time.time()
surface_reflectivity_lw = 0.05
helper_dim = "rhshape"
for i, rh in enumerate(lw_fluxes[helper_dim].values):
    prof = lw_fluxes.sel({helper_dim: rh})
    surface_temp = prof.sel(altitude=0)["ta"].item()

    result = flux_simulator.flux_simulator_single_profile(
        atms_grd[i],
        surface_temp,
        0.0,
        surface_reflectivity=surface_reflectivity_lw,
        z_field=prof["altitude"],
    )

    # Store spectral and integrated fluxes
    lw_fluxes["lw_flux_up_spectral"].loc[{helper_dim: rh}] = result[
        "spectral_flux_clearsky_up"
    ].T
    lw_fluxes["lw_flux_down_spectral"].loc[{helper_dim: rh}] = result[
        "spectral_flux_clearsky_down"
    ].T
    lw_fluxes["lw_flux_up"].loc[{helper_dim: rh}] = result["flux_clearsky_up"]
    lw_fluxes["lw_flux_down"].loc[{helper_dim: rh}] = result["flux_clearsky_down"]
    lw_fluxes["heating_rate"].loc[{helper_dim: rh}] = result["heating_rate_clearsky"]

    # ETA logging
    elapsed = time.time() - start_time
    remaining = elapsed / (i + 1) * (lw_fluxes[helper_dim].size - i - 1)
    print(
        f"{i + 1}/{lw_fluxes[helper_dim].size} complete | ETA: {remaining:.1f}s (~{remaining / 60:.1f} min)",
        flush=True,
        )


lw_fluxes.to_zarr(
    "/scratch/m/m301046/artsfluxes_ideal.zarr",
    mode="w",
)