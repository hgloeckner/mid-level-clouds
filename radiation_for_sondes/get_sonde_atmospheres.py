# %%
import numpy as np
import xarray as xr
import typhon.physics
import pyarts.recipe
import time

from pydropsonde.helper.xarray_helper import write_ds

ipfs_gateway = "https://ipfs.io"
ipfs_hash = "bafybeicb33v6ohezyhgq5rumq4g7ejnfqxzcpuzd4i2fxnlos5d7ebmi3m"
pressure_levels = 16
sonde_alt = 15e3  # 15 km

# %%

pyarts.data.download()
raw_data = xr.open_dataset(f"{ipfs_gateway}/ipfs/{ipfs_hash}", engine="zarr")

valid_data_mask = (raw_data.ta_qc == 0) & (raw_data.rh_qc == 0) & (raw_data.p_qc == 0)
interpolated_data = (
    raw_data[["sonde_id", "p", "rh", "ta"]]
    .where(valid_data_mask, drop=True)
    .interpolate_na(dim="altitude", method="akima")
    .interpolate_na(
        dim="altitude", method="linear", fill_value="extrapolate", max_gap=300
    )
)

# %%
vmr = xr.apply_ufunc(
    typhon.physics.relative_humidity2vmr,
    interpolated_data.rh,
    interpolated_data.p,
    interpolated_data.ta,
    output_dtypes=[float],
)
clean_data = interpolated_data.assign(vmr=vmr).sortby("altitude")

# %%
start_time = time.time()
atmospheres = []
for sonde in clean_data.sonde.values:
    ds = clean_data.sel(sonde=sonde)

    sfc_temp = ds.ta.isel(altitude=0).values
    flux_strato = pyarts.recipe.AtmosphericFlux(
        surface_temperature=sfc_temp, max_level_step=1000.0
    )
    flux_tropos = pyarts.recipe.AtmosphericFlux(
        surface_temperature=sfc_temp, max_level_step=200.0
    )
    strato_atm = flux_strato.get_atmosphere()
    tropos_atm = flux_tropos.get_atmosphere()

    new_atm = {}
    for variable in tropos_atm.keys():
        new_atm[variable] = np.concat(
            [strato_atm[variable][:36], tropos_atm[variable][-70:]]
        )

    new_atm["t"][-70:] = ds.ta.sortby("altitude", ascending=False).values[-1400::20]
    new_atm["p"][-70:] = ds.p.sortby("altitude", ascending=False).values[-1400::20]
    new_atm["H2O"][-70:] = ds.vmr.sortby("altitude", ascending=False).values[-1400::20]
    new_atm["CO2"][:] = 0.00042  # 420 ppm

    atmospheres.append(
        xr.Dataset(
            data_vars={var: (("altitude",), new_atm[var]) for var in new_atm.keys()},
            coords={
                "sonde_id": (("sonde",), [str(ds.sonde_id.values)]),
            },
        )
    )
    elapsed = time.time() - start_time
    remaining = elapsed / (sonde + 1) * (clean_data.sonde.size - sonde - 1)
    print(
        f"{sonde + 1}/{clean_data.sonde.size} complete | ETA: {remaining:.1f}s (~{remaining / 60:.1f} min)"
    )

# %%


output_path = "idealized_atmospheres.zarr"
write_ds(
    xr.concat(atmospheres, dim="sonde").interpolate_na(dim="altitude", method="akima"),
    dir=".",
    filename=output_path,
    object_dims=("sonde",),
    alt_dim="altitude",
)
# %%
ds = xr.open_dataset(output_path, engine="zarr")
# %%
for sonde in ds.sonde:
    ds.sel(sonde=sonde).t.plot()
