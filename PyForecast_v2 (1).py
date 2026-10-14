
import os
from datetime import datetime

import requests
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np


# --------------------------------------------------
# AWPC MODEL COMPARISON METEOGRAMS
# --------------------------------------------------

LAT = 39.87
LON = -75.24
STATION = "KPHL"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Model-specific endpoints and optional model parameters
MODELS = {
    "GFS": {
        "url": "https://api.open-meteo.com/v1/forecast",
        "params": {"models": "gfs_seamless"},
    },
    "ICON": {
        "url": "https://api.open-meteo.com/v1/dwd-icon",
        "params": {},
    },
    "ECMWF": {
        "url": "https://api.open-meteo.com/v1/ecmwf",
        "params": {},
    },
    "UKMO": {
        "url": "https://api.open-meteo.com/v1/forecast",
        "params": {"models": "ukmo_seamless"},
    },
    "GEM": {
        "url": "https://api.open-meteo.com/v1/gem",
        "params": {},
    },
}

VARIABLES = {
    "precipitation": {
        "name": "Precipitation",
        "field": "precipitation",
        "unit": "inch",
        "extra": {"precipitation_unit": "inch"},
        "label": "Precipitation (in/hr)",
        "filename": "meteogram_precipitation.png",
    },
    "temperature": {
        "name": "Temperature",
        "field": "temperature_2m",
        "unit": "°F",
        "extra": {"temperature_unit": "fahrenheit"},
        "label": "Temperature (°F)",
        "filename": "meteogram_temperature.png",
    },
    "wind": {
        "name": "Wind Speed",
        "field": "wind_speed_10m",
        "unit": "kt",
        "extra": {"windspeed_unit": "kn"},
        "label": "Wind Speed (kt)",
        "filename": "meteogram_wind.png",
    },
    "pressure": {
        "name": "Sea Level Pressure",
        "field": "pressure_msl",
        "unit": "hPa",
        "extra": {},
        "label": "MSLP (hPa)",
        "filename": "meteogram_pressure.png",
    },
}


def fetch_model(model_name, model, variable):
    """Fetch one model, preserving missing observations as NaN."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": variable["field"],
        "forecast_days": 3,
        "timezone": "UTC",
        **model["params"],
        **variable["extra"],
    }

    response = requests.get(
        model["url"],
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]
    times = [
        datetime.fromisoformat(t)
        for t in hourly["time"]
    ]

    values = [
        np.nan if value is None else float(value)
        for value in hourly[variable["field"]]
    ]

    return dict(zip(times, values))


def get_metar():
    """Retrieve a recent KPHL observation for chart context."""
    try:
        response = requests.get(
            "https://aviationweather.gov/api/data/metar",
            params={"ids": STATION, "format": "raw"},
            timeout=15,
        )
        response.raise_for_status()
        return response.text.strip().splitlines()[0]
    except (requests.RequestException, IndexError):
        return "Latest KPHL METAR unavailable"


def make_chart(variable_key, variable, metar):
    print(f"\nFetching {variable['name']} forecasts...", flush=True)

    # Each model is independent: one failure won't stop the others.
    model_data = {}

    for name, model in MODELS.items():
        try:
            model_data[name] = fetch_model(name, model, variable)
            print(
                f"  {name}: {len(model_data[name])} hourly values",
                flush=True,
            )
        except Exception as exc:
            print(f"  {name}: unavailable ({exc})", flush=True)

    if not model_data:
        raise RuntimeError("No models returned usable forecast data.")

    # Align every model by actual timestamp, not array position.
    all_times = sorted({
        timestamp
        for values in model_data.values()
        for timestamp in values
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, values in model_data.items():
        y = [values.get(t, np.nan) for t in all_times]
        ax.plot(
            all_times,
            y,
            linewidth=1.4,
            marker=".",
            markersize=2,
            label=name,
        )

    # Average only the models that have valid data at each hour.
    ensemble_mean = []

    for timestamp in all_times:
        valid = [
            values[timestamp]
            for values in model_data.values()
            if timestamp in values
            and np.isfinite(values[timestamp])
        ]

        ensemble_mean.append(
            float(np.mean(valid)) if valid else np.nan
        )

    ax.plot(
        all_times,
        ensemble_mean,
        color="black",
        linewidth=2.8,
        label="Ensemble Mean",
        zorder=10,
    )

    ax.xaxis.set_major_locator(
        mdates.HourLocator(byhour=[0, 6, 12, 18], tz=mdates.UTC)
    )
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, pos: (
                mdates.num2date(x, tz=mdates.UTC).strftime("%HZ\n%b %d")
                if mdates.num2date(x, tz=mdates.UTC).hour == 0
                else mdates.num2date(x, tz=mdates.UTC).strftime("%HZ")
            )
        )
    )

    plt.setp(
        ax.get_xticklabels(),
        fontsize=8,
        rotation=0,
        ha="center",
    )

    ax.set_title(
        f"{variable['name']} Model Comparison Meteogram\n"
        f"KPHL area | {datetime.now().strftime('%Y-%m-%d')} | UTC"
    )
    ax.set_ylabel(variable["label"])
    ax.set_xlabel("Time (UTC)")
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.legend(fontsize=8, ncol=2, loc="best")

    # Small observation reference without dumping the whole METAR.
    fig.text(
        0.02, 0.01,
        f"Latest KPHL METAR: {metar[:110]}",
        fontsize=7,
        color="dimgray",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    output_path = os.path.join(PLOT_DIR, variable["filename"])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {output_path}", flush=True)


def main():
    print("=== AWPC Model Comparison Meteograms ===", flush=True)
    print(f"Forecast location: {LAT}, {LON}", flush=True)

    metar = get_metar()

    for key, variable in VARIABLES.items():
        try:
            make_chart(key, variable, metar)
        except Exception as exc:
            print(
                f"FAILED: {variable['name']}: {exc}",
                flush=True,
            )

    print("\nMeteogram generation finished.", flush=True)


if __name__ == "__main__":
    main()
