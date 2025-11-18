# %%
import xarray as xr
from orcestra import get_flight_segments
import myutils.data_helper as dh
import myutils.open_datasets as od

# %%

lam_sondes = xr.open_dataset(
    "/scratch/m/m301046/lam_sondes_z.zarr",
    engine="zarr",
)
# %%
cids = od.get_cids()
beach = od.open_dropsondes(cids["dropsondes"])
lev4 = xr.open_dataset(
    f"ipfs://{cids['orcestra']}/products/HALO/dropsondes/Level_4/PERCUSION_Level_4.zarr",
    engine="zarr",
)

# %%
lam_l4_sondes = lam_sondes.swap_dims({"sonde": "sonde_id"}).sel(
    sonde_id=lev4.sonde_id.values
)

# %% calculate circle products
