import numpy as np
import moist_thermodynamics.constants as constants


def get_freezing_levels(ds, vars=None):
    if vars is None:
        vars = ["rh", "u", "v", "tv", "mse", "vmse", "umse"]

    swap_d = ds.swap_dims({"sonde": "sonde_id"})
    indices = (
        (np.abs(swap_d.ta.interpolate_na(dim="altitude") - 273.15))
        .dropna(dim="sonde_id", how="all")
        .argmin(dim="altitude")
    )

    for variable in vars:
        d = []
        varlist = []
        for sonde_id in swap_d.sonde_id:
            try:
                alt = ds.altitude[indices.sel(sonde_id=sonde_id)].values
                d.append(alt)
            except KeyError:
                d.append(np.nan)
                varlist.append(np.nan)
            else:
                varlist.append(swap_d.sel(altitude=alt, sonde_id=sonde_id)[variable])
        swap_d = swap_d.assign({f"freezing_{variable}": (("sonde_id"), varlist)})
    swap_d = swap_d.assign(freezing_level=(("sonde_id"), d))
    return swap_d.swap_dims({"sonde_id": "sonde"})


def calc_Tv(T, mr):
    """
    Calculate the virtual temperature (Tv) from temperature (T) and mixing ratio (mr).
    Tv = T * (1 + 0.61 * mr)
    """
    eps = constants.eps1  # kg / kg
    return T * (1 + mr / eps) / (1 + mr)
