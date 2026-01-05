import numpy as np
import pandas as pd


def datetime_to_day(time):
    return pd.to_datetime(time).timetuple().tm_yday


def declination_angle_model(d_n):
    # return in rad
    a_n = -0.399912
    b_n = 0.070257
    theta = 2 * np.pi * d_n / 365
    return a_n * np.cos(theta) + b_n * np.sin(theta)


def get_hour_from_time(time):
    dtime = pd.to_datetime(time)
    hour = dtime.hour
    minute = dtime.minute
    return hour + minute / 60.0


def get_h(time):
    # get hour angle from datetime64 time
    hour = get_hour_from_time(time)
    assert np.all(hour >= 0) and np.all(hour < 24)
    return 15 * (12 - hour)


def get_local_time(time, lon):
    return time + (np.deg2rad(lon) / np.pi * 12).astype("timedelta64[h]")


def cos_zenith_angle(time, lat, lon):
    local_time = get_local_time(np.datetime64(time, "ns"), lon)
    declination = declination_angle_model(datetime_to_day(local_time))
    h = np.deg2rad(get_h(local_time))
    lat = np.deg2rad(lat)
    mu = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(
        h
    )
    return mu


def get_mu_day(date, lat, lon):
    hourlist = [date + np.timedelta64(h, "h") for h in range(0, 24)]

    return [cos_zenith_angle(t, lat, lon) for t in hourlist]
