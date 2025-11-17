# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append("../")
from myutils import open_datasets
from myutils import physics_helper as physics
from myutils.data_helper import sel_sub_domain
# %%

wv, no_wv = open_datasets.open_wales(masked=True)
wv = wv.assign(q=physics.wv2q(wv))

cid = "ipns://latest.orcestra-campaign.org"  # open_datasets.get_cid()
dropsondes = xr.open_dataset(
    f"{cid}/products/HALO/dropsondes/Level_3/PERCUSION_Level_3.zarr",
    engine="zarr",
)


# %%
def find_highest_cloud_altitude(ds, variable_name="bsrgl", threshold=20):
    ds = ds.sortby("altitude")  # .chunk({"altitude": -1, "time": 1000})
    mask = ds[variable_name] >= threshold
    mask_inv = mask.isel(altitude=slice(None, None, -1))

    highest_altitude = ds.altitude.values[
        mask.sizes["altitude"] - 1 - mask_inv.argmax(dim="altitude")
    ]
    return np.where(mask.any(dim="altitude"), highest_altitude, np.nan)


cloud_top = xr.DataArray(
    data=find_highest_cloud_altitude(no_wv), dims="sonde", name="cloud_top"
).to_dataframe()
# %%

east = [[-34.5, 2.5], [-34.5, 13.5], [-20, 13.5], [-20, 2.5]]
west = [[-59, 6], [-59, 17], [-44.5, 17], [-44.5, 6]]
north = [[-26, 13.5], [-26, 19], [-20, 19], [-20, 13.5]]
cloud_top_north = xr.DataArray(
    data=find_highest_cloud_altitude(
        sel_sub_domain(
            no_wv, north, item_var="time", lon_var="longitude", lat_var="latitude"
        )
    ),
    dims="sonde",
    name="cloud_top",
).to_dataframe()
cloud_top_east = xr.DataArray(
    data=find_highest_cloud_altitude(
        sel_sub_domain(
            no_wv, east, item_var="time", lon_var="longitude", lat_var="latitude"
        )
    ),
    dims="sonde",
    name="cloud_top",
).to_dataframe()
cloud_top_west = xr.DataArray(
    data=find_highest_cloud_altitude(
        sel_sub_domain(
            no_wv, west, item_var="time", lon_var="longitude", lat_var="latitude"
        )
    ),
    dims="sonde",
    name="cloud_top",
).to_dataframe()
# %%
sns.set_context("paper", font_scale=1.3)
fig, ax = plt.subplots(figsize=(8, 5))
histkwargs = {
    "stat": "density",
    "bins": 30,
    "binrange": (0, 14000),
    "element": "step",
    "fill": False,
    "linewidth": 1,
    "kde": False,
    "alpha": 0.5,
    "ax": ax,
    "y": "cloud_top",
}
# sns.histplot(cloud_top,color="k", **histkwargs)
# sns.histplot(cloud_top_north, color="#FF7982", **histkwargs)
# sns.histplot(cloud_top_east, color="#B6001E",  **histkwargs)
# sns.histplot(cloud_top_west, color="#00b4d8",  **histkwargs)

kdekwargs = {
    "fill": False,
    "linewidth": 3,
    "ax": ax,
    "y": "cloud_top",
}
pt = sns.kdeplot(cloud_top, color="k", **kdekwargs)
sns.kdeplot(cloud_top_north, color="#FF7982", label="North", **kdekwargs)
sns.kdeplot(cloud_top_east, color="#B6001E", label="East", **kdekwargs)
sns.kdeplot(cloud_top_west, color="#00b4d8", label="West", **kdekwargs)

ax.axhline(5836.80, xmax=0.5, color="k", alpha=0.5)
ax.axhline(6559.80, xmax=0.5, color="#FF7982", alpha=0.5)
ax.axhline(5819.43, xmax=0.5, color="#B6001E", alpha=0.5)
ax.axhline(5378.91, xmax=0.5, color="#00b4d8", alpha=0.5)
ax.set_yticks([0, 2000, 4000, 5380, 5830, 6560, 8000, 10000, 12000, 14000])
sns.despine()
ax.legend()
ax.set_ylim(0, 14000)
ax.set_ylabel("Cloud Top Altitude / m")
fig.tight_layout()
fig.savefig("plots/cloud_top_altitude_distribution.pdf")
fig.savefig("/scratch/m/m301046/cloud_top_altitude_distribution.pdf", transparent=True)
# %%
for i in range(4):
    x, y = pt.lines[4 + i].get_data()
    idx4000 = np.argmin(np.abs(y - 4000))
    idx8000 = np.argmin(np.abs(y - 9000))
    midmax = np.argmax(x[idx4000:idx8000]) + idx4000
    print(f"Max mid level cloud top altitude for region {i}: {y[midmax]:.2f} m")
# %%
for region, height in zip(
    ["north", "east", "west"],
    [6559.80, 5819.43, 5378.91],
):
    ds = sel_sub_domain(
        dropsondes,
        eval(region),
        item_var="sonde",
        lon_var="launch_lon",
        lat_var="launch_lat",
    )
    print(
        "ct temperature for",
        region,
        ": ",
        ds.ta.sel(altitude=height, method="nearest").mean("sonde").values,
        "K",
    )
