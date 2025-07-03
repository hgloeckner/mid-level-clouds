#%%
import os
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import pyarts
import FluxSimulator as fsm

#%%
min_wn = 1
max_wn = 3000
wave_bands = 10000
# %% import example atmosphere

idealized_sondes = "/work/mh0066/m301046/ml_clouds/combined_atmospheres.nc"

atmosphere = xr.open_dataset(idealized_sondes)
# %% import example atmosphere
atmosphere2 = xr.open_dataset("~/code/fun_with_arts/data/atms.nc")
test_ds = xr.open_dataset("/work/mh0066/m301046/ml_clouds/sondes_for_radiation.nc")
# %% convert xarray to ARTS gridded field
atms_grd = pyarts.arts.ArrayOfGriddedField4()
ds = atmosphere.swap_dims({"sonde_id":"sonde"})
for i in range(ds.sonde.size):
    profile = ds.isel(sonde=i)
    
    profile_grd = fsm.generate_gridded_field_from_profiles(
        profile["p"].values.astype(float),
        profile["t"].values,
        gases={
            "H2O": profile["H2O"],
            "CO2": profile["CO2"],
            "O3": profile["O3"],
            "N2": profile["N2"],
            "O2": profile["O2"],
        },
        z_field=profile["altitude"].values
    )
    atms_grd.append(profile_grd)
#%%
gases = ["H2O", "CO2", "O2", "N2", "O3"]
f_grid = np.linspace(min_wn, max_wn, wave_bands)  # frequency grid in cm^-1
f_grid_freq = pyarts.arts.convert.kaycm2freq(f_grid)  # converted to Hz
surface_reflectivity_lw = 0.05

species = [
    "H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
    "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
    "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
    "CO2, CO2-CKDMT252",
    "O3",
    "O3-XFIT",
]

flux_simulator = fsm.FluxSimulator("beachsondes")
flux_simulator.ws.f_grid = f_grid_freq
flux_simulator.set_species(species)
#%%
flux_simulator.get_lookuptableBatch(atms_grd)
#%%
# Initialize output datasets
shape_flux = (len(ds.sonde), len(ds.altitude), len(f_grid))
shape_integrated = (len(ds.sonde), len(ds.altitude))

lw_fluxes = xr.Dataset(
    {
        "lw_flux_up_spectral": (("sonde", "altitude", "f_grid"), np.zeros(shape_flux)),
        "lw_flux_down_spectral": (("sonde", "altitude", "f_grid"), np.zeros(shape_flux)),
        "lw_flux_up": (("sonde", "altitude"), np.zeros(shape_integrated)),
        "lw_flux_down": (("sonde", "altitude"), np.zeros(shape_integrated)),
        "heating_rate": (("sonde", "altitude"), np.zeros(shape_integrated)),
    },
    coords={"sonde": ds.sonde, "altitude": ds.altitude, "f_grid": f_grid}
)
#%%
# Run simulation and store results
start_time = time.time()

for i in range(ds.sonde.size):
    prof = ds.isel(sonde=i)
    surface_temp = prof.isel(altitude=0)["t"].item()
    
    result = flux_simulator.flux_simulator_single_profile(
        atms_grd[i],
        surface_temp,
        0.,
        surface_reflectivity=surface_reflectivity_lw,
        z_field=prof["altitude"]
    )

    # Store spectral and integrated fluxes
    lw_fluxes["lw_flux_up_spectral"].loc[dict(sonde=i)] = result["spectral_flux_clearsky_up"].T
    lw_fluxes["lw_flux_down_spectral"].loc[dict(sonde=i)] = result["spectral_flux_clearsky_down"].T
    lw_fluxes["lw_flux_up"].loc[dict(sonde=i)] = result["flux_clearsky_up"]
    lw_fluxes["lw_flux_down"].loc[dict(sonde=i)] = result["flux_clearsky_down"]
    lw_fluxes["heating_rate"].loc[dict(sonde=i)] = result["heating_rate_clearsky"]

    # ETA logging
    elapsed = time.time() - start_time
    remaining = elapsed / (i + 1) * (ds.sonde.size - i - 1)
    print(f"{i + 1}/{ds.sonde.size} complete | ETA: {remaining:.1f}s (~{remaining / 60:.1f} min)")

#%%

from pydropsonde.helper.xarray_helper import write_ds
write_ds(
    lw_fluxes,,
    dir="/work/mh0066/m301046/ml_clouds/",
    filename="arts2_fluxes.nc",
    object_dims=("sonde",),
    alt_dim="altitude",
)
