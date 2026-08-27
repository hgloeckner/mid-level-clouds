
import xarray as xr
import myutils.physics_helper as ph
import numcodecs

gases = {
    "co2": 422e-6,
    "ch4": 1650e-9,
    "n2o": 306e-9,
    "n2": 0.78084,
    "o2": 0.20946,
    "co": 0.0,
}

lw_reflectivity = 0.05
sw_reflectivity = 0.06

def make_rrtmg_atm(profile):
    return xr.Dataset(
        data_vars={
            "pres_level": (("column", "level"), profile["p"].values),
            "temp_level": (("column", "level"), profile["ta"].values),
            "pres_layer": (("column", "layer"), 0.5 * (profile["p"].values[..., 1:] + profile["p"].values[...,:-1])),
            "temp_layer": (("column", "layer"), 0.5 * (profile["ta"].values[..., 1:] + profile["ta"].values[...,:-1])),
            "surface_temperature": (("column",), profile["ta"].sel(altitude=0, method="nearest").values),
            "h2o": (("column", "layer"), 0.5 * (ph.specific_humidity2vmr(profile["q"].values[..., 1:]) + ph.specific_humidity2vmr(profile["q"].values[..., :-1]))),
            "o3": (("column", "layer"), 0.5 * (profile["o3"].values[..., 1:] + profile["o3"].values[...,:-1])),
            "co2": gases["co2"],
            "ch4": gases["ch4"],
            "n2o": gases["n2o"],
            "n2": gases["n2"],
            "o2": gases["o2"],
            "co": gases["co"],
        },
        coords={
           "launch_lat": (("column"),profile.launch_lat.values),
            "launch_lon": (("column"),profile.launch_lon.values),
            "launch_time": (("column"),profile.launch_time.values),

        }
    )


def get_chunks(sizes, chunksize=393216):
    sonde_chunksize = 1
    match tuple(sizes.keys()):
        case ("sonde", "altitude", "f_grid"):
            chunks = {
                "sonde": sonde_chunksize,
                "f_grid": chunksize // (10 * sonde_chunksize),
                "altitude": 10,
            }
        case ("sonde", "altitude", "f_grid_sw"):
            chunks = {
                "sonde": sonde_chunksize,
                "f_grid_sw": chunksize // (10 * sonde_chunksize),
                "altitude": 10,
            }
        case ("sonde", "altitude", "f_grid_lw"):
            chunks = {
                "sonde": sonde_chunksize,
                "f_grid_lw": chunksize // (10 * sonde_chunksize),
                "altitude": 10,
            }
        case ("mu0","sonde", "altitude"):
            chunks = {
                "mu0": -1,
                "sonde": sonde_chunksize,
                "altitude": -1,
                            }
        case ("sonde", "altitude", "hour_of_day"):
                    chunks = {
                        "hour_of_day": -1,
                        "sonde": sonde_chunksize,
                        "altitude": -1,
                                    }
        case ("sonde", "altitude", "f_grid_sw", "hour_of_day"):
                            sonde_chunksize = 5
                            chunks = {
                                "hour_of_day": -1,
                                "sonde": sonde_chunksize,
                                "f_grid_sw": chunksize // (10 * sonde_chunksize),
                                "altitude": 10,
                                            }
        case ("sonde", "level"):
            chunks = {
                "sonde": sonde_chunksize,
                "level": -1,
            }
        case ("sonde", "layer"):
            chunks = {
                "sonde": sonde_chunksize,
                "layer": -1,
            }

        case ("sonde", "altitude"):
            chunks = {
                "sonde": sonde_chunksize,
                "altitude": -1,
            }
        case (single_dim,):
            chunks = {
                single_dim: -1,
            }
        case _:
            chunks = {}

    return tuple((chunks[d] for d in sizes))


def get_encoding(dataset):
    compressor = numcodecs.Blosc("zstd", shuffle=2, clevel=6)

    return {
        var: {
            "compressor": compressor,
            "dtype": "float32",
            "chunks": get_chunks(dataset[[var]].sizes),
        }
        for var in dataset.variables
        if (var != "sonde_id") and (var not in dataset.coords)
    }


def bitround(ds, keepbits=16, codec=None):
    def _bitround(var, keepbits, codec=None):
        if codec is None:
            codec = numcodecs.BitRound(keepbits=keepbits)

        return codec.decode(codec.encode(var))

    ds_rounded = xr.apply_ufunc(
        _bitround,
        ds,
        kwargs={"keepbits": keepbits},
        keep_attrs=True,
        dask="parallelized",
    )

    return ds_rounded
