"""Fetch 2024 hourly weather for the grid points covering the selected stops.

9 grid points at 0.1 degree resolution cover all 380 stops used by the 8 selected
bus lines. One Open-Meteo Archive API call per grid point for the full year,
cached permanently to data/processed/weather.parquet so the pipeline never needs
to call the API again.
"""
import time
import duckdb
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

HOURLY = ["temperature_2m", "precipitation", "rain", "snowfall",
          "snow_depth", "weather_code", "wind_speed_10m", "cloud_cover"]

WX_GROUPS = {
    0: "clear", 1: "cloudy", 2: "cloudy", 3: "cloudy",
    45: "fog", 48: "fog",
    51: "drizzle", 53: "drizzle", 55: "drizzle", 56: "drizzle", 57: "drizzle",
    61: "rain", 63: "rain", 65: "rain", 66: "rain", 67: "rain",
    71: "snow", 73: "snow", 75: "snow", 77: "snow",
    80: "rain_showers", 81: "rain_showers", 82: "rain_showers",
    85: "snow_showers", 86: "snow_showers",
    95: "thunderstorm", 96: "thunderstorm", 99: "thunderstorm",
}


def get_grid_points(con):
    rows = con.sql("""
        WITH s AS (SELECT DISTINCT observed_stop_id FROM 'data/processed/headways.parquet')
        SELECT DISTINCT round(st.stop_lat,1) grid_lat, round(st.stop_lon,1) grid_lon
        FROM s LEFT JOIN read_csv_auto('data/raw/stops.csv') st ON s.observed_stop_id = st.stop_id
        ORDER BY 1,2
    """).fetchall()
    return rows


def fetch_grid_point(om, lat, lon):
    r = om.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params={"latitude": lat, "longitude": lon,
                "start_date": "2024-01-01", "end_date": "2024-12-31",
                "hourly": HOURLY, "wind_speed_unit": "ms"},
    )[0]
    h = r.Hourly()
    out = {"ts_utc": pd.date_range(
        start=pd.to_datetime(h.Time(), unit="s", utc=True),
        end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=h.Interval()), inclusive="left")}
    for i, name in enumerate(HOURLY):
        out[name] = h.Variables(i).ValuesAsNumpy()
    df = pd.DataFrame(out)
    df["grid_lat"], df["grid_lon"] = lat, lon
    return df


def main():
    con = duckdb.connect()
    grid_points = get_grid_points(con)
    print(f"{len(grid_points)} grid points: {grid_points}")

    session = retry(requests_cache.CachedSession("data/processed/.wxcache", expire_after=-1),
                     retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=session)

    t0 = time.time()
    frames = [fetch_grid_point(om, lat, lon) for lat, lon in grid_points]
    wx = pd.concat(frames, ignore_index=True)
    print(f"fetched {len(wx)} rows in {time.time()-t0:.1f}s")

    wx = wx.sort_values(["ts_utc"]).reset_index(drop=True)
    wx["ts_utc"] = wx["ts_utc"].dt.tz_localize(None)  # store naive UTC, tz implicit

    wx["is_precip"] = wx["precipitation"] > 0.1
    wx["precip_3h"] = wx.groupby(["grid_lat", "grid_lon"])["precipitation"].transform(
        lambda s: s.rolling(3, min_periods=1).sum())
    wx["is_snowing"] = wx["snowfall"] > 0
    wx["ice_risk"] = (wx["temperature_2m"] < 1) & (wx["precipitation"] > 0)
    wx["temp_bin"] = pd.cut(wx["temperature_2m"], bins=[-100, -5, 0, 5, 15, 25, 100],
                             labels=["<-5", "-5_0", "0_5", "5_15", "15_25", ">25"])
    wx["wx_group"] = wx["weather_code"].round().astype("Int64").map(WX_GROUPS).fillna("other")

    wx.to_parquet("data/processed/weather.parquet", index=False)
    print("wrote data/processed/weather.parquet:", wx.shape)
    print(wx.head())

    n_hours_expected = len(grid_points) * 366 * 24
    print(f"rows vs expected full-year hourly coverage: {len(wx)} / {n_hours_expected}")


if __name__ == "__main__":
    main()
