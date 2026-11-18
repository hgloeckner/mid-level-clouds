# %%
import xarray as xr
import numpy as np
import orcestra
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xhistogram.xarray import histogram
import myutils.open_datasets as od
import myutils.data_helper as dh
# %%


lev4 = ds = xr.open_dataset(
    "ipfs://bafybeibgeeqs5uhmbqy4hz4v3pihrfisiklcetisxkf63d2r473szaprwi", engine="zarr"
).swap_dims({"circle": "circle_id"})
lev3 = od.open_dropsondes("bafybeiczbv7mycr2jois6t4dq3zwiltycomwo5xxvjqcjz2ot3newzar6q")
radarct = xr.open_dataset("/Users/helene/Documents/scratch/" + "radar_max_cloud_top.nc")
# %%
meta = orcestra.get_flight_segments()
segments = [
    {**s, "platform_id": platform_id, "flight_id": flight_id}
    for platform_id, flights in meta.items()
    for flight_id, flight in flights.items()
    for s in flight["segments"]
    if np.isin(s["segment_id"], lev4.circle_id.values)
]
# %%
radarcircles = []
fractions = []

for segment in segments:
    newtime = pd.date_range(
        start=segment["start"],
        end=segment["end"],
        freq="1s",
    )
    circleds = (
        radarct.sel(time_idx=slice(segment["start"], segment["end"]))
        .reindex(time_idx=newtime, method="nearest", tolerance="1s")
        .dropna(dim="time_idx")
    )
    circlefill = circleds.reindex(time_idx=newtime, method="nearest", tolerance="1s")
    print(
        radarct.sel(time_idx=slice(segment["start"], segment["end"])).sizes["time_idx"]
    )
    print(circleds.sizes["time_idx"])
    time_size = circlefill.sizes["time_idx"]
    time_cld_size = circleds.sizes["time_idx"]
    fraction = xr.Dataset(
        dict(
            high_frac=(
                "circle_id",
                [
                    (
                        circlefill.cloud_top.where(
                            circlefill.cloud_top > 8000, drop=True
                        ).sizes["time_idx"]
                        / time_size
                    )
                ],
            ),
            high_cld_frac=(
                "circle_id",
                [
                    (
                        circleds.cloud_top.where(
                            circleds.cloud_top > 8000, drop=True
                        ).sizes["time_idx"]
                        / time_cld_size
                    )
                ],
            ),
            mid_frac=(
                "circle_id",
                [
                    (
                        circlefill.cloud_top.where(
                            (circlefill.cloud_top <= 8000)
                            & (circlefill.cloud_top > 4000),
                            drop=True,
                        ).sizes["time_idx"]
                        / time_size
                    )
                ],
            ),
            mid_cld_frac=(
                "circle_id",
                [
                    (
                        circleds.cloud_top.where(
                            (circleds.cloud_top <= 8000) & (circleds.cloud_top > 4000),
                            drop=True,
                        ).sizes["time_idx"]
                        / time_cld_size
                    )
                ],
            ),
            low_frac=(
                "circle_id",
                [
                    (
                        circlefill.cloud_top.where(
                            circlefill.cloud_top <= 4000, drop=True
                        ).sizes["time_idx"]
                        / time_size
                    )
                ],
            ),
            low_cld_frac=(
                "circle_id",
                [
                    (
                        circleds.cloud_top.where(
                            circleds.cloud_top <= 4000, drop=True
                        ).sizes["time_idx"]
                        / time_cld_size
                    )
                ],
            ),
            no_frac=(
                "circle_id",
                [
                    (
                        circlefill.cloud_top.where(
                            circlefill.cloud_top.isnull(), drop=True
                        ).sizes["time_idx"]
                        / time_size
                    )
                ],
            ),
            meanct=("circle_id", [(circleds.cloud_top.mean("time_idx").values)]),
        ),
    )
    print(
        np.sum(
            [
                fraction.high_frac.values,
                fraction.mid_frac.values,
                fraction.low_frac.values,
                fraction.no_frac.values,
            ]
        )
    )
    print(
        np.sum(
            [
                fraction.high_cld_frac.values,
                fraction.mid_cld_frac.values,
                fraction.low_cld_frac.values,
            ]
        )
    )

    fractions.append(fraction.assign_coords({"circle_id": [segment["segment_id"]]}))
    radarcircles.append(circleds)

maxct = xr.concat(radarcircles, dim="time_idx")
maxctfract = xr.concat(fractions, dim="circle_id")
maxctfract = maxctfract.assign(low_no_frac=maxctfract.low_frac + maxctfract.no_frac)
# %%

cloudtype = {
    key: maxctfract.where(
        (maxctfract[key + "_frac"] > 0.1)
        & (maxctfract.meanct > minct)
        & (maxctfract.meanct < maxct),
        drop=True,
    )
    for key, minct, maxct in [
        ("high", 8000, 17000),
        ("mid", 4000, 8000),
        ("low_no", 0, 4000),
    ]
}


# %%
sns.set_palette("tab10")
fig, ax = plt.subplots()
for key, idx in cloudtype.items():
    sids = idx.circle_id.values
    print(key, sids.shape)
    lev4.sel(circle_id=sids).mean("circle_id").omega.plot(
        y="altitude", ax=ax, label=key
    )
lev4.omega.mean("circle_id").plot(y="altitude", ax=ax, label="all", color="k")
ax.axvline(0, color="grey", lw=1, ls="-")
ax.legend()
sns.despine()
# %%

# %%


fig, ax = plt.subplots()
for subdomain in [
    (dh.east, "#960018", "East"),
    (dh.west, "#0085db", "West"),
    (dh.north, "#FFA500", "North"),
    (dh.full, "black", "Full"),
]:
    ds = lev3.pipe(dh.sel_sub_domain, subdomain[0])
    ds = ds.where(~np.isnan(ds.rh), drop=True)
    hist = histogram(
        ds.ta,
        bins=[np.linspace(210, 310, 100)],
        dim=["altitude"],
        weights=ds.rh,
    ) / histogram(
        ds.ta,
        bins=[np.linspace(210, 310, 100)],
        dim=["altitude"],
    )
    (100 * hist).mean("sonde").plot(
        y="ta_bin", ax=ax, color=subdomain[1], label=subdomain[2]
    )
    (100 * hist).quantile(0.1, dim="sonde").plot(
        y="ta_bin", ax=ax, color=subdomain[1], ls="--", alpha=0.5
    )
    (100 * hist).quantile(0.9, dim="sonde").plot(
        y="ta_bin", ax=ax, color=subdomain[1], ls="--", alpha=0.5
    )
sns.despine()
ax.set_title("")
ax.legend()
ax.invert_yaxis()
ax.set_xlabel("Relative Humidity / %")
ax.set_ylabel("$T$ / K")
# ax.set_ylim(0, 12000)
# %%
var = "iwv_mean"
fig, ax = plt.subplots()
for subdomain in [(dh.east, "#960018", "East"), (dh.west, "#0085db", "West")]:
    ds = lev4[["iwv_mean", "omega", "rh_mean"]].pipe(
        dh.sel_sub_domain,
        subdomain[0],
        "circle_id",
        lat_var="circle_lat",
        lon_var="circle_lon",
    )

    dry = ds[var].quantile(0.33, dim="circle_id")
    moist = ds[var].quantile(0.66, dim="circle_id")

    (ds.where(ds[var] > moist) * 60 * 60 / 100).omega.mean("circle_id").plot(
        y="altitude", color=subdomain[1], alpha=0.5
    )
    (ds.where(ds[var] < dry) * 60 * 60 / 100).omega.mean("circle_id").plot(
        y="altitude", color=subdomain[1], alpha=0.5
    )
    (ds.omega.mean("circle_id") * 60 * 60 / 100).plot(
        y="altitude", color=subdomain[1], label=subdomain[2]
    )
ax.axvline(0, color="grey", lw=1, ls="-")
ax.set_ylim(0, 12000)
ax.legend()
ax.set_xlabel("$\omega$ / hPa h$^{-1}$")
ax.set_ylabel("altitude / m")
sns.despine()
