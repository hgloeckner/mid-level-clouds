# %%

import rad_helper as rad
import matplotlib.pyplot as plt
from pyrte_rrtmgp.rrtmgp_data_files import (
    CloudOpticsFiles,
    GasOpticsFiles,
)
from pyrte_rrtmgp.examples import (
    compute_RCE_profiles,
)
from pyrte_rrtmgp.rrtmgp import GasOptics, CloudOptics

# %%
cloud_optics_lw = CloudOptics(cloud_optics_file=CloudOpticsFiles.LW_BND)
gas_optics_lw = GasOptics(gas_optics_file=GasOpticsFiles.LW_G256)

cloud_optics_sw = CloudOptics(cloud_optics_file=CloudOpticsFiles.SW_BND)
gas_optics_sw = GasOptics(gas_optics_file=GasOpticsFiles.SW_G224)


# %%
def make_profiles(ncol=24, nlay=72):
    # Create atmospheric profiles and gas concentrations
    atmosphere = compute_RCE_profiles(300, ncol, nlay)

    # Add other gas values
    gas_values = {
        "co2": 348e-6,
        "ch4": 1650e-9,
        "n2o": 306e-9,
        "n2": 0.7808,
        "o2": 0.2095,
        "co": 0.0,
    }

    for gas_name, value in gas_values.items():
        atmosphere[gas_name] = value

    return atmosphere


atmosphere = make_profiles()
# %%
optical_props = gas_optics_lw.compute(
    atmosphere,
    add_to_input=False,
)
optical_props["surface_emissivity"] = 0.98
clr_fluxes = optical_props.rte.solve(add_to_input=False)
clr_fluxes
# %%
plt.plot(clr_fluxes.lw_flux_up.isel(column=0), clr_fluxes.level, label="Flux up")
plt.plot(clr_fluxes.lw_flux_down.isel(column=0), clr_fluxes.level, label="Flux down")
plt.legend(frameon=False)

# %%

plt.plot(
    rad.calc_heating_rate_from_flx(
        flx_up=clr_fluxes.lw_flux_up.isel(column=0),
        flx_down=clr_fluxes.lw_flux_down.isel(column=0),
        p=atmosphere.pres_level.isel(column=0),
    )
    * 60
    * 60
    * 24,
    atmosphere.pres_level.isel(column=0) / 100,
)
plt.xlim(-5, 5)
plt.gca().invert_yaxis()
# %%
optical_props = gas_optics_sw.compute(
    atmosphere,
    add_to_input=False,
)

optical_props["surface_albedo"] = 0.06
optical_props["mu0"] = 1
fluxes = optical_props.rte.solve(add_to_input=False)

# %%
plt.plot(
    rad.calc_heating_rate_from_flx(
        flx_up=clr_fluxes.lw_flux_up.isel(column=0),
        flx_down=clr_fluxes.lw_flux_down.isel(column=0),
        p=atmosphere.pres_level.isel(column=0),
    )
    * 60
    * 60
    * 24,
    atmosphere.pres_level.isel(column=0) / 100,
)

plt.plot(
    rad.calc_heating_rate_from_flx(
        flx_up=fluxes.sw_flux_up.isel(column=0),
        flx_down=fluxes.sw_flux_down.isel(column=0),
        p=atmosphere.pres_level.isel(column=0),
    )
    * 60
    * 60
    * 24,
    atmosphere.pres_level.isel(column=0) / 100,
)

plt.plot(
    rad.calc_heating_rate_from_flx(
        flx_up=(clr_fluxes.lw_flux_up + fluxes.sw_flux_up).isel(column=0),
        flx_down=(clr_fluxes.lw_flux_down + fluxes.sw_flux_down).isel(column=0),
        p=atmosphere.pres_level.isel(column=0),
    )
    * 60
    * 60
    * 24,
    atmosphere.pres_level.isel(column=0) / 100,
)

plt.xlim(-5, 5)
plt.gca().invert_yaxis()
# %%
