import numpy as np
import pandas as pd
import pytz
import timezonefinder


def datetime_to_day(time):
    return time.timetuple().tm_yday


def declination_angle_model(d_n):
    # return in rad
    a_n = -0.399912
    b_n = 0.070257
    theta = 2 * np.pi * d_n / 365
    return a_n * np.cos(theta) + b_n * np.sin(theta)


def get_hour_from_time(time):
    hour = time.hour
    minute = time.minute
    return hour + minute / 60.0


def get_h(time):
    # get hour angle from datetime64 time
    hour = get_hour_from_time(time)
    assert hour >= 0 and hour < 24
    return 15 * (12 - hour)


def get_local_time(time, lat, lon):
    timezone = timezonefinder.TimezoneFinder().timezone_at(lat=lat, lng=lon)
    tz = pytz.timezone(timezone)
    return pd.to_datetime(time) + tz.utcoffset(pd.to_datetime(time))


def cos_zenith_angle(time, lat, lon):
    print(time)
    local_time = get_local_time(time, lat, lon)
    print(local_time)
    declination = declination_angle_model(datetime_to_day(local_time))
    print("declination", declination)
    print("h", get_h(local_time))
    h = np.deg2rad(get_h(local_time))
    lat = np.deg2rad(lat)
    print("cos h", np.cos(h))
    print("cos lat", np.cos(lat))
    print("sin lat", np.sin(lat))
    mu = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(
        h
    )
    print(mu)
    return mu
