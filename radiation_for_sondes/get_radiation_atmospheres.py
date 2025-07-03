# %%
import xarray as xr
import numpy as np
import typhon
from pydropsonde.helper.xarray_helper import write_ds


ipfs_gateway = "https://ipfs.io"
ipfs_hash = "bafybeicb33v6ohezyhgq5rumq4g7ejnfqxzcpuzd4i2fxnlos5d7ebmi3m"
tropo_step = 100
# %%
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
arts3_atm = xr.open_dataset("/work/mh0066/m301046/ml_clouds/idealized_atmospheres.nc")

coarsened = (
    clean_data.sel(altitude=slice(1, None))
    .coarsen(altitude=tropo_step // 10, boundary="pad")
    .mean("altitude")
)
sondes_to_tropo = xr.concat(
    [clean_data.sel(altitude=0), coarsened.assign(altitude=coarsened.altitude - 5)],
    dim="altitude",
).assign(sonde_id=clean_data.sonde_id)

# %%
atmospheres = []
for sonde_id in arts3_atm.sonde_id.values:
    atm = arts3_atm.swap_dims({"sonde": "sonde_id"}).sel(sonde_id=sonde_id)
    new_vals = (
        sondes_to_tropo.swap_dims({"sonde": "sonde_id"})
        .sel(sonde_id=sonde_id)
        .rename({"vmr": "H2O", "ta": "t"})
    )
    atmospheres.append(
        xr.concat(
            [
                new_vals,
                atm.sel(altitude=slice(new_vals.altitude.max() + tropo_step, None)),
            ],
            dim="altitude",
        )
    )

# %%
ds_for_radiation = (
    xr.concat(atmospheres, dim="sonde_id")
    .assign(
        {
            var: (
                ("altitude"),
                atm[var]
                .interp_like(atmospheres[-1], kwargs={"fill_value": "extrapolate"})
                .values,
            )
            for var in ["O2", "O3", "N2", "CO2"]
        }
    )
    .interpolate_na("altitude")
)
# %%

write_ds(
    ds_for_radiation,
    dir="/work/mh0066/m301046/ml_clouds/",
    filename="combined_atmospheres.nc",
    object_dims=("sonde_id",),
    alt_dim="altitude",
)
