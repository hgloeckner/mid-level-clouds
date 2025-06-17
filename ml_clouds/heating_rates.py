#%%
import xarray as xr
from xhistogram.xarray import histogram
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo
#%%
flux_data_arts2 = xr.open_dataset("/work/mh0066/m301046/ml_clouds/arts2_fluxes.zarr", engine="zarr")
#%%
hist_heating_rate = histogram(
        flux_data_arts2.t.sel(altitude=slice(0, 14000)),
        (flux_data_arts2.heating_rate.sel(altitude=slice(0, 14000))*(-1)) ,
        bins=[np.linspace(220, 300, 100), np.linspace(0, 15, 100)],
    )

#%%
small_size = 10
medium_size = 12
bigger_size = 14
plt.rc('font', size=small_size)          # controls default text sizes
plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=small_size)    # legend fontsize
plt.rc('figure', titlesize=bigger_size) 
linewidth = 2


fig,ax = plt.subplots(figsize=(5, 5))
hist_heating_rate.plot(cmap="cmo.ice_r", add_colorbar=False)
((hist_heating_rate * hist_heating_rate.heating_rate_bin).sum(dim="heating_rate_bin") / 
    (hist_heating_rate.sum(dim="heating_rate_bin"))).plot(y = "t_bin", color="black",
                                                          linewidth=linewidth, ax=ax)

ax.invert_yaxis()
ax.set_ylabel("Temperature / K")
ax.set_xlabel("Cooling Rate / K day-1")
ax.axhline(273.15, color="black", linestyle="--")
ax.set_xlim(0, 5)
ax.set_ylim(300, 250)
fig.savefig("images/heating_rates_sondes_low.pdf", bbox_inches="tight")