# %%
import numpy as np
import xarray as xr
import typhon.physics
import pyarts.recipe
import time

from pydropsonde.helper.xarray_helper import write_ds

ipfs_gateway = "https://ipfs.io"
ipfs_hash = "bafybeicb33v6ohezyhgq5rumq4g7ejnfqxzcpuzd4i2fxnlos5d7ebmi3m"
strato_step = 1000
tropo_step = 100
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
def create_fictional_ds(atm_dict, sonde_id):
    return xr.Dataset(
        data_vars={
            var: (("altitude",), atm_dict[var]) for var in ["t", "p", "H2O", "O3"]
        },
        coords={
            "sonde_id": (("sonde",), [sonde_id]),
            "altitude": (("altitude",), atm_dict["altitude"]),
        },
    )


start_time = time.time()
atmospheres = []
for sonde in clean_data.sonde.values:
    ds = clean_data.sel(sonde=sonde)

    sfc_temp = ds.ta.isel(altitude=0).values
    flux_strato = pyarts.recipe.AtmosphericFlux(
        surface_temperature=sfc_temp, max_level_step=strato_step
    )
    flux_tropos = pyarts.recipe.AtmosphericFlux(
        surface_temperature=sfc_temp, max_level_step=tropo_step
    )
    strato_atm = flux_strato.get_atmosphere()
    strato_atm["altitude"] = np.arange(0, 50000 + strato_step, strato_step)[::-1] + (
        strato_step // 2
    )
    tropos_atm = flux_tropos.get_atmosphere()
    tropos_atm["altitude"] = np.arange(0, 50000 + tropo_step, tropo_step)[::-1] + (
        tropo_step // 2
    )

    tropos_ds = create_fictional_ds(tropos_atm, str(ds.sonde_id.values)).sortby(
        "altitude"
    )
    strato_ds = create_fictional_ds(strato_atm, str(ds.sonde_id.values)).sortby(
        "altitude"
    )
    new_atm = xr.concat(
        [
            tropos_ds.sel(altitude=slice(None, 20000)),
            strato_ds.sel(altitude=slice(20000, None)),
        ],
        dim="altitude",
    )
    atmospheres.append(new_atm)
    elapsed = time.time() - start_time
    remaining = elapsed / (sonde + 1) * (clean_data.sonde.size - sonde - 1)
    print(
        f"{sonde + 1}/{clean_data.sonde.size} complete | ETA: {remaining:.1f}s (~{remaining / 60:.1f} min)"
    )
ds = xr.concat(atmospheres, dim="sonde").assign(
    CO2=(("altitude",), np.full_like(atmospheres[0].altitude, 420e-6, dtype="float32")),
    O2=(("altitude",), np.full_like(atmospheres[0].altitude, 0.209, dtype="float32")),
    N2=(("altitude",), np.full_like(atmospheres[0].altitude, 0.781, dtype="float32")),
    O3=(("altitude",), atmospheres[0].O3.values),
)
# %%
from pydropsonde.helper.xarray_helper import write_ds

write_ds(
    ds,
    dir="/work/mh0066/m301046/ml_clouds/",
    filename="idealized_atmospheres.nc",
    object_dims=("sonde",),
    alt_dim="altitude",
)

# %%

arts3 = xr.open_dataset("/work/mh0066/m301046/ml_clouds/idealized_atmospheres.nc")
# %%

output_path = "idealized_atmospheres.zarr"
write_ds(
    xr.concat(atmospheres, dim="sonde").interpolate_na(dim="altitude", method="akima"),
    dir="/work/mh0066/m301046/ml_clouds/",
    filename=output_path,
    object_dims=("sonde",),
    alt_dim="altitude",
)
# %%
ds = xr.open_dataset(output_path, engine="zarr")
# %%
for sonde in ds.sonde:
    ds.sel(sonde=sonde).t.plot()
