#%%

import numpy as np
import mlclouds.myutils.data_helper as datautils
import xarray as xr
import ssl, certifi 
import moist_thermodynamics.constants as mtc
#%%


beach = (xr.open_dataset("ipfs://bafybeiczbv7mycr2jois6t4dq3zwiltycomwo5xxvjqcjz2ot3newzar6q", engine="zarr")
         .pipe(datautils.interpolate_gaps).pipe(datautils.extrapolate_sfc)
         
         )
beach = xr.concat(
    [beach.sel(altitude=0),
    beach.coarsen(altitude=5, boundary="trim").mean()
    ],
    dim="altitude"  
)[["ta", "q", "p", "rh"]]

#%%
ssl_context = ssl.create_default_context(cafile=certifi.where())
era5 = xr.open_dataset(
    "https://data.earthdatahub.destine.eu/era5/era5-pressure-levels-v0.zarr",
    storage_options={"client_kwargs":{"trust_env":True}, "ssl": ssl_context},
    chunks={},
    engine="zarr",
).rename(
                  {"isobaricInhPa":"p",
                   "t":"ta",
                   "r":"rh",}
)
era5 = era5.assign(
    rh = era5.rh / 100,
    p = era5.p * 100,
)
#%%          

def merge_ds(sonde, era5):
    era5 = era5.assign(
            altitude=(mtc.radius_earth * era5.z / (mtc.radius_earth - era5.z))/ mtc.gravity_earth,
    ).swap_dims({"p":"altitude"}).reset_coords(["p"])
    sondeoverlap = sonde.sel(altitude=slice(12000, 14000)).coarsen(altitude=2, boundary="trim").mean()
    eraoverlap = era5.interp(altitude=sondeoverlap.altitude)
    high_alts = np.concat(
        [np.linspace(14000, 20000, 7),
        np.linspace(25000, 50000, 6)]
    )
    new_sonde = xr.merge(
        [
            xr.concat(
                [ 
                    sonde.sel(altitude=slice(0, 12000))[var],
                    (sondeoverlap[var] * (14000 - sondeoverlap.altitude) / 2000)
                    + (eraoverlap[var] * (sondeoverlap.altitude - 12000) / 2000),
                    era5.interp(altitude=high_alts)[var]  
                ],
            dim="altitude"
            )
            for var in ["ta", "q", "p"]
        ]
    )
    new_sonde = new_sonde.assign(
        q = xr.where(new_sonde.altitude >= 25000, 4e-8 , xr.where(new_sonde.altitude < 20000, new_sonde.q, np.nan)).chunk({"altitude":-1}).interpolate_na(dim="altitude")
    )
    return xr.merge(
        [new_sonde,
         era5.o3.interp(altitude=new_sonde.altitude)
        ]
    ).chunk({"altitude":-1}).interpolate_na(dim="altitude", fill_value="extrapolate").assign_coords({"altitude":new_sonde.altitude})

merged_profiles =[]
for sondeid in beach.sonde:
    print(sondeid.values)
    sonde = beach.isel(sonde=sondeid)
    era5_sonde = era5.sel(valid_time=sonde.launch_time,latitude=sonde.launch_lat, longitude=sonde.launch_lon, method="nearest")
    full_profile = merge_ds(sonde, era5_sonde)
    
    merged_profiles.append(full_profile)
#%%

#full_profile.to_netcdf("/scratch/m/m301046/test_profile.nc", mode="w")
#%%
ds = xr.concat(merged_profiles, dim="sonde")


#%%
ds.to_netcdf("/scratch/m/m301046/radiation_profiles.nc")
