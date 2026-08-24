#%%

import numpy as np
import xarray as xr
import pyarts
from pyarts.arts import convert
import FluxSimulator as fsm
import myutils.physics_helper as ph
from pyrte_rrtmgp.rrtmgp import GasOptics
from pyrte_rrtmgp.rrtmgp.data_files import (
    GasOpticsFiles,
)
import rad_helper as rh

profile = xr.open_dataset("/scratch/m/m301046/test_profile.nc")#sondes.isel(sonde=100)
profile.ta.plot()

#%% ARTS
min_wavelength_sw = 3e-7  # [m]
max_wavelength_sw = 5e-6  # [m]
n_freq_sw = 200
surface_reflectivity_sw = rh.sw_reflectivity
surface_reflectivity_lw = rh.lw_reflectivity

wvl = np.linspace(min_wavelength_sw, max_wavelength_sw, n_freq_sw)  # [m]
f_grid_sw = convert.wavelen2freq(wvl[::-1])  # [Hz]

min_wvn = 10  # [cm^-1]
max_wvn = 3210  # [cm^-1]
n_freq_lw = 200
wvn = np.linspace(min_wvn, max_wvn, n_freq_lw)
f_grid_lw = convert.kaycm2freq(wvn)

lat = profile.launch_lat.values
lon = profile.launch_lon.values
surface_altitude= 0.0  # [m]
surface_temp = profile.ta.sel(altitude=0, method="nearest").values
# sun position



sun_pos = ph.get_arts_sun_pos(profile.launch_time.values)

atms_grd = pyarts.arts.ArrayOfGriddedField4()

#%%
profile_grd = fsm.generate_gridded_field_from_profiles(
        profile["p"].values,
        profile["ta"].values,
        gases={
            "H2O": ph.specific_humidity2vmr(profile["q"]),
            "CO2": np.repeat(rh.gases["co2"], profile["q"].shape),
            "O3": profile["o3"],
            "N2": np.repeat(rh.gases["n2"], profile["q"].shape),
            "O2": np.repeat(rh.gases["o2"], profile["q"].shape),
            "CH4": np.repeat(rh.gases["ch4"], profile["q"].shape),
        },
        z_field=profile["altitude"].values,
    )

LW_flux_simulator = fsm.FluxSimulator("beach"+ "_LW")
#%%
species = [
            "H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
            "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
            "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
            "CO2, CO2-CKDMT252",
            "O3",
            "O3-XFIT",
        ]
LW_flux_simulator.ws.f_grid = f_grid_lw
LW_flux_simulator.set_species(species)
atms_grd.append(profile_grd)
LW_flux_simulator.get_lookuptableBatch(atms_grd)
#%%
lw = LW_flux_simulator.flux_simulator_single_profile(
        profile_grd,
        surface_temp,
        surface_altitude,
        surface_reflectivity_lw,
        geographical_position=[lat, lon],
    )

#%%
SW_flux_simulator = fsm.FluxSimulator("beach" + "_SW")
SW_flux_simulator.ws.f_grid = f_grid_sw
SW_flux_simulator.emission = 0
SW_flux_simulator.gas_scattering = True
SW_flux_simulator.set_sun(sun_pos)
sw = SW_flux_simulator.flux_simulator_single_profile(
    profile_grd,
    surface_temp,
    surface_altitude,
    surface_reflectivity_sw,
    geographical_position=[lat, lon],
)

#%% RRTMG 
profile = profile.expand_dims(["column"]).assign(
    launch_lat=(("column"), [profile.launch_lat.values]),
    launch_lon=(("column"), [profile.launch_lon.values]),
    launch_time=(("column"), [profile.launch_time.values]),
)

rrtmg_atm = rh.make_rrtmg_atm(profile)
# %%



gas_optics_lw = GasOptics(gas_optics_file=GasOpticsFiles.LW_G256)
gas_optics_sw = GasOptics(gas_optics_file=GasOpticsFiles.SW_G224)
optical_props = gas_optics_lw.compute(rrtmg_atm, add_to_input=False)

optical_props = optical_props.assign(surface_emissivity=0.98)
# %%
lw_fluxes = optical_props.rte.solve(add_to_input=False)
optical_props_sw = gas_optics_sw.compute(rrtmg_atm, add_to_input=False)

optical_props_sw["surface_albedo"] = 0.06
optical_props_sw = optical_props_sw.assign(mu0=("column", [1]))
sw_fluxes = optical_props_sw.rte.solve(add_to_input=False)
#%%

import myutils.physics_helper as ph
import matplotlib.pyplot as plt

plt.plot(
    ph.calc_heating_rate_from_flx(
        sw["flux_clearsky_up"],
        -sw["flux_clearsky_down"],
        sw["pressure"]
    )*60*60*24, 
    profile.altitude.values,
)
plt.plot(
    sw["heating_rate_clearsky"].squeeze(),
    profile.altitude.values,
)
plt.plot(
    ph.calc_heating_rate_from_flx(
        sw_fluxes["sw_flux_up"].squeeze(),
        sw_fluxes["sw_flux_down"].squeeze(),
        profile["p"].values
    )*60*60*24,
    sw["altitude"]
)
plt.ylim(0, 10000)