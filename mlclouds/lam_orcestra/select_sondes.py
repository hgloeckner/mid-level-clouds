# %%
import xarray as xr
import glob
import numpy as np
import healpix

import myutils.data_helper as dh
import easygems.healpix as egh
import myutils.open_datasets as od
from pydropsonde.helper import xarray_helper as xh

# %%
path_to_files = glob.glob(
    "/work/mh0492/m301067/orcestra/healpix/**-rerun/orcestra_1250m_*-rerun_3d_hpz12.zarr"
)
lam_3d = xr.open_mfdataset(
    path_to_files,
    engine="zarr",
    parallel=True,
    chunks={"time": 12, "height_full": -1, "cell": 16384},
).pipe(egh.attach_coords)
# %%
_, unique_idx = np.unique(lam_3d.time, return_index=True)
lam_3d = lam_3d.isel(time=unique_idx)
# %%

cids = od.get_cids()
beach = (
    od.open_dropsondes(cids["dropsondes"])
    .pipe(dh.interpolate_gaps)
    .pipe(dh.extrapolate_sfc)
    .sel(sonde=slice(2, None))
)
rapsodi = (
    od.open_radiosondes(cids["radiosondes"])
    .pipe(dh.interpolate_gaps)
    .pipe(dh.extrapolate_sfc)
)
# %%


def get_lat_lon_mask(dx, lat, lon):
    return healpix.ang2pix(
        egh.get_nside(dx), lon % 360, lat, nest=egh.get_nest(dx), lonlat=True
    )


def sel_sonde(dx, sonde):
    icell = get_lat_lon_mask(
        dx, sonde.launch_lat.values % 180, sonde.launch_lon.values % 360
    )
    return dx.sel(cell=int(icell)).sel(time=sonde.launch_time.values, method="nearest")


lam_profiles = []
for sonde in beach.sonde:
    lam_profiles.append(
        sel_sonde(lam_3d, beach.isel(sonde=sonde)).assign(
            sonde_id=beach.sonde_id.isel(sonde=sonde)
        )
    )
for sonde in rapsodi.sonde:
    lam_profiles.append(
        sel_sonde(lam_3d, rapsodi.isel(sonde=sonde)).assign(
            sonde_id=rapsodi.set_coords(["launch_lat", "launch_lon"]).sonde_id.isel(
                sonde=sonde
            )
        )
    )

# %%
lam_sonde = xr.concat(lam_profiles, dim="sonde")  # .assign_coords(sonde=beach.sonde)
# %%

grid = xr.open_dataset(
    "/work/mh0492/m301067/orcestra/auxiliary-files/grids/ORCESTRA_1250m_DOM01_vgrid.nc"
)
heights = (
    grid.where((np.rad2deg(grid.clon) > -40))
    .where((np.rad2deg(grid.clon) < -30))
    .where((np.rad2deg(grid.clat) < 10))
    .where((np.rad2deg(grid.clat) > 8))
    .mean("ncells")
)

# %%
lam_complete = lam_sonde.assign(
    z=heights.zg.rename({"height_2": "height_full"}),
    z_half=heights.zghalf.rename({"height": "height_half"}),
).swap_dims({"height_full": "z", "height_half": "z_half"})
# %%
lam_complete = lam_complete.assign(
    w=(
        ("sonde", "z"),
        lam_complete.wa.interp(z_half=lam_complete.z).values,
        lam_complete.wa.attrs,
    )
).drop_vars(["wa", "height_half", "z_half"])
# %%

xh.write_ds(
    lam_complete.reset_coords(["height_full"]).chunk({"sonde": -1, "z": -1}),
    "/scratch/m/m301046/",
    "lam_sondes_z.zarr",
    object_dims=("sonde",),
    alt_dim="z",
)

# %%
# %% select precip from IMERG for sondes.

imerg_dir = "/pool/data/ICDC/atmosphere/imerg/DATA/2024/"
aug_list = glob.glob(imerg_dir + "IMERG_precipitationrate*202408*.nc")
sep_list = glob.glob(imerg_dir + "IMERG_precipitationrate*202409*.nc")
file_list = [f for f in aug_list + sep_list]
ds = xr.open_mfdataset(file_list)
# %%
imerg_sondes = []
for sonde in beach.sonde:
    bs = beach.isel(sonde=sonde)
    imerg_sonde = ds.sel(
        time=bs.launch_time.values,
        lat=bs.launch_lat.values,
        lon=bs.launch_lon.values,
        method="nearest",
    )
    imerg_sondes.append(imerg_sonde.assign(sonde_id=bs.sonde_id))
# %%
imerg_sondes = xr.concat(imerg_sondes, dim="sonde_id").load()
# %%
imerg_sondes.chunk({"sonde_id": -1}).to_zarr(
    "/scratch/m/m301046/imerg_sondes.zarr", mode="w"
)

# %%
imerg_area = ds.sel(lat=slice(-8, 27.5), lon=slice(-68, -2)).sel(
    time=lam_3d.time, method="nearest"
)
# %%
imerg_area.chunk({"time": 200, "lat": 200, "lon": 600}).to_zarr(
    "/scratch/m/m301046/imerg_lam.zarr", mode="w"
)
# %%


ifs_sondes = xr.open_dataset(
    "/work/mh0492/m301067/orcestra/results/timeseries/ifs_interpolated_on_dropsondes_profiles_2nd-days.nc"
)

# %%
lam_2d = xr.open_mfdataset(
    glob.glob(
        "/work/mh0492/m301067/orcestra/healpix/**-rerun/orcestra_1250m_*-rerun_2d_hpz12.zarr"
    ),
    engine="zarr",
    parallel=True,
)
# %%
lam2d = []
for sonde in lam_sonde.sonde_id.values:
    lam3d = lam_sonde.swap_dims({"sonde": "sonde_id"}).sel(sonde_id=sonde)
    lam2d.append(
        lam_2d.sel(time=lam3d.time, cell=lam3d.cell).assign_coords(sonde_id=sonde)
    )
lam_2d_sondes = xr.concat(lam2d, dim="sonde_id")
# %%
lam_2d_sondes.isel(time=0).to_zarr(
    "/scratch/m/m301046/lam_sondes_2d.zarr", mode="w", zarr_format=2
)
