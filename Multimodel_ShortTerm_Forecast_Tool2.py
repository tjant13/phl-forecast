import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from scipy.interpolate import make_interp_spline

# ---------------------------------------------------------------------------

# Model catalog (flat list, no high-res/low-res grouping)

# ---------------------------------------------------------------------------

MODELS = {

    "1": ("NAM 3km (CONUS Nest)", "ncep_nam_conus"),
    "2": ("HRRR", "ncep_hrrr_conus"),
    "3": ("GFS", "ncep_gfs_global"),
    "4": ("ECMWF IFS", "ecmwf_ifs025"),
    "5": ("ICON Global", "dwd_icon_global"),
    "6": ("ICON EU", "dwd_icon_eu"),
    "7": ("ICON D2 (high-res nest)", "dwd_icon_d2"),
    "8": ("GEM (Canadian)", "cmc_gem_seamless"),
    "9": ("UKMO", "ukmo_seamless"),
    "10": ("Meteo-France (ARPEGE/AROME)", "meteofrance_seamless"),
    "11": ("JMA (Japan)", "jma_seamless"),
    "12": ("BOM ACCESS (Australia)", "bom_access_global"),

}



# used for hours/models with no data (model doesn't cover this location)

NODATA_COLOR = "#bdbdbd"


# ===========================================================================
# VARIABLE 1: PRECIPITATION
# ===========================================================================

PRECIP_COLORS = {

    ("rain", "light"): "#8bdd8b",
    ("rain", "moderate"): "#4caf50",
    ("rain", "heavy"): "#2f8735",
    ("snow", "light"): "#8ebfd5",
    ("snow", "moderate"): "#63afd8",
    ("snow", "heavy"): "#3375a7",
    ("mix", "light"): "#cb9fe1",
    ("mix", "moderate"): "#c179d9",
    ("mix", "heavy"): "#9532b3",
}

# mm/hr thresholds for intensity buckets

PRECIP_LIGHT_MAX = 0.5
PRECIP_MODERATE_MAX = 2.5

# A model contributes more to the score the more intense its forecast is at
# that hour.

PRECIP_WEIGHTS = {"light": 3, "moderate": 4, "heavy": 5}

def classify_precip(precip, rain, snow):

    """Values arrive in the order of the variable's 'fields' list."""

    if precip is None:

        return "nodata"

    if precip <= 0:

        return None

    rain = rain or 0

    snow = snow or 0

    if snow > 0 and rain <= 0:

        ptype = "snow"

    elif snow > 0 and rain > 0:

        ptype = "mix"

    else:

        ptype = "rain"



    if precip < PRECIP_LIGHT_MAX:

        intensity = "light"

    elif precip < PRECIP_MODERATE_MAX:

        intensity = "moderate"

    else:

        intensity = "heavy"

    return ptype, PRECIP_COLORS[(ptype, intensity)], PRECIP_WEIGHTS[intensity]





# ===========================================================================

# VARIABLE 2: WIND GUSTS (mph)

# ===========================================================================

# (lower bound in mph, legend label, color, support weight), weakest first.

# A gust falls in the highest tier whose lower bound it meets or exceeds.

GUST_TIERS = [

    (18, "Gusts 18+ mph", "#fbf4a8", 1),    # light yellow

    (25, "Gusts 25+ mph", "#f5e050", 1),    # yellow

    (35, "Gusts 35+ mph", "#f9b866", 1),    # light orange

    (45, "Gusts 45+ mph", "#f2794d", 1),    # light scarlet / reddish orange

    (60, "Gusts 60+ mph", "#e8514f", 1),    # red / light red

    (80, "Gusts 80+ mph", "#e0508f", 1),    # reddish pink

    (100, "Gusts 100+ mph", "#e44dc4", 1),  # pink

]





def classify_gust(gust):

    if gust is None:

        return "nodata"

    tier = None

    for i, (lower, _, _, _) in enumerate(GUST_TIERS):

        if gust >= lower:

            tier = i

    if tier is None:

        return None  # below the lowest threshold

    _, _, color, weight = GUST_TIERS[tier]

    return tier, color, weight





# ===========================================================================

# VARIABLE 3: CAPE (J/kg)

# ===========================================================================

# (lower bound in J/kg, legend label, color, support weight), weakest first.

# Color scale runs light red -> dark red -> medium pinks -> magenta at the

# extreme end. A value falls in the highest tier whose lower bound it meets.

CAPE_TIERS = [

    (250, "CAPE 250+", "#f9bdb5", 1),    # light red

    (500, "CAPE 500+", "#f38b82", 1),    # salmon red

    (1000, "CAPE 1000+", "#e5534b", 1),  # red

    (1500, "CAPE 1500+", "#c92a2d", 1),  # strong red

    (2000, "CAPE 2000+", "#9f1a24", 1),  # dark red

    (3000, "CAPE 3000+", "#d1477e", 1),  # medium pink

    (4000, "CAPE 4000+", "#e6479b", 1),  # hot pink

    (5000, "CAPE 5000+", "#e83fc5", 1),  # magenta

]





def classify_cape(cape):

    if cape is None:

        return "nodata"

    tier = None

    for i, (lower, _, _, _) in enumerate(CAPE_TIERS):

        if cape >= lower:

            tier = i

    if tier is None:

        return None  # below the lowest threshold

    _, _, color, weight = CAPE_TIERS[tier]

    return tier, color, weight





# ===========================================================================

# VARIABLE 4: TEMPERATURE (F)

# ===========================================================================

# 5 F steps from -20 F up to 110 F. Colors are blended between the anchor

# colors below, so every 5-degree step gets its own shade:

#   very dark red at 110+, dark red at 100s, red at 90s, red-orange at 80s,

#   orange at 70s, yellow at 60s, greens in the 40s-50s, blues/indigo in the

#   cold range, purple at -20.

TEMP_MIN = -20

TEMP_MAX = 110

TEMP_STEP = 5



TEMP_ANCHORS = [

    (-20, "#5b2a86"),  # purple

    (-10, "#4a3fb0"),  # indigo

    (0,   "#2f62c2"),  # deep blue

    (10,  "#3b8fd9"),  # blue

    (20,  "#5ec2e6"),  # light blue

    (30,  "#3fb5a8"),  # teal

    (40,  "#6cc07a"),  # green

    (50,  "#c5de6a"),  # yellow-green

    (60,  "#f5d84a"),  # yellow

    (70,  "#f9a03f"),  # orange

    (80,  "#f2682d"),  # red-orange

    (90,  "#e03a2f"),  # red

    (100, "#a50f15"),  # dark red

    (110, "#6b0010"),  # very dark red

]





def _build_temp_tiers():

    """Return [(lower_bound_F, legend_label, hex_color, weight), ...], coldest

    first, one entry per TEMP_STEP, colors interpolated between the anchors."""

    tiers = []

    for t in range(TEMP_MIN, TEMP_MAX + TEMP_STEP, TEMP_STEP):

        for (t0, c0), (t1, c1) in zip(TEMP_ANCHORS, TEMP_ANCHORS[1:]):

            if t0 <= t <= t1:

                f = (t - t0) / (t1 - t0)

                rgb0, rgb1 = mcolors.to_rgb(c0), mcolors.to_rgb(c1)

                rgb = tuple(a + (b - a) * f for a, b in zip(rgb0, rgb1))

                color = mcolors.to_hex(rgb)

                break

        if t == TEMP_MIN:

            label = f"{t}°F & below"

        else:

            label = f"{t}°F+"

        tiers.append((t, label, color, 1))

    return tiers





TEMP_TIERS = _build_temp_tiers()



# Daily high/low labels on each model bar. A point is labeled when it is the

# highest (or lowest) value within +/- this many hours of itself...

EXTREMA_WINDOW_HOURS = 5

# ...and the temperature in that window swings by at least this many F, so a

# flat or noisy stretch doesn't get a label.

EXTREMA_MIN_SWING_F = 3





def classify_temp(temp):

    """Temperature always has a value, so every hour gets a color. Anything

    colder than the lowest tier is clamped into it."""

    if temp is None:

        return "nodata"

    tier = 0

    for i, (lower, _, _, _) in enumerate(TEMP_TIERS):

        if temp >= lower:

            tier = i

    return tier, TEMP_TIERS[tier][2], 1





# ===========================================================================

# VARIABLE REGISTRY - the menu, API request, charts, and legend all read this

# ===========================================================================

# Keys in each entry:

#   name          menu text

#   title         used in the chart title

#   slug          used in the output filename

#   fields        Open-Meteo hourly field names; their values are passed to

#                 classify() in this same order

#   api_extra     extra API params (e.g. unit overrides)

#   classify      see "HOW TO ADD A NEW VARIABLE" at the top of this file

#   bottom_chart  "support" (default, stacked support chart) or "lines"

#                 (one line per model + bold black average; uses the first

#                 field's raw values)

#   layers        support chart only: stacked layers, bottom -> top

#                 (layer_key, fill_color)

#   max_weight    support chart only: highest weight classify can return

#   support_label support chart: y-axis label for the stacked chart

#   line_label    lines chart: y-axis label

#   legend        list of (label, color) entries for the legend

#   legend_ncol   number of legend columns

VARIABLES = {

    "1": {

        "name": "Precipitation",

        "title": "Precipitation",

        "slug": "precip",

        "fields": ["precipitation", "rain", "snowfall"],

        "api_extra": {},

        "classify": classify_precip,

        "bottom_chart": "support",

        "layers": [

            ("rain", PRECIP_COLORS[("rain", "moderate")]),

            ("mix", PRECIP_COLORS[("mix", "moderate")]),

            ("snow", PRECIP_COLORS[("snow", "moderate")]),

        ],

        "max_weight": max(PRECIP_WEIGHTS.values()),

        "support_label": "Precip support\n(weighted %)",

        "legend": [

            ("Light Rain", PRECIP_COLORS[("rain", "light")]),

            ("Mod. Rain", PRECIP_COLORS[("rain", "moderate")]),

            ("Heavy Rain", PRECIP_COLORS[("rain", "heavy")]),

            ("Light Snow", PRECIP_COLORS[("snow", "light")]),

            ("Mod. Snow", PRECIP_COLORS[("snow", "moderate")]),

            ("Heavy Snow", PRECIP_COLORS[("snow", "heavy")]),

            ("Light Mix", PRECIP_COLORS[("mix", "light")]),

            ("Heavy Mix", PRECIP_COLORS[("mix", "heavy")]),

            ("No Data", NODATA_COLOR),

            ("Rain Probability", PRECIP_COLORS[("rain", "moderate")]),

            ("Mix Probability", PRECIP_COLORS[("mix", "moderate")]),

            ("Snow Probability", PRECIP_COLORS[("snow", "moderate")]),

        ],

        "legend_ncol": 4,

    },

    "2": {

        "name": "Wind gusts",

        "title": "Wind Gust",

        "slug": "wind_gust",

        "fields": ["wind_gusts_10m"],

        "api_extra": {"wind_speed_unit": "mph"},

        "classify": classify_gust,

        "bottom_chart": "support",

        "layers": [(i, color) for i, (_, _, color, _) in enumerate(GUST_TIERS)],

        "max_weight": max(t[3] for t in GUST_TIERS),

        "support_label": "Gust support\n(weighted %)",

        "legend": [(label, color) for _, label, color, _ in GUST_TIERS]

                  + [("No Data", NODATA_COLOR)],

        "legend_ncol": 8,

    },

    "3": {

        "name": "CAPE",

        "title": "CAPE",

        "slug": "cape",

        "fields": ["cape"],

        "api_extra": {},

        "classify": classify_cape,

        "bottom_chart": "support",

        "layers": [(i, color) for i, (_, _, color, _) in enumerate(CAPE_TIERS)],

        "max_weight": max(t[3] for t in CAPE_TIERS),

        "support_label": "CAPE support\n(weighted %)",

        "legend": [(label, color) for _, label, color, _ in CAPE_TIERS]

                  + [("No Data", NODATA_COLOR)],

        "legend_ncol": 5,

    },

    "4": {

        "name": "Temperature",

        "title": "Temperature",

        "slug": "temperature",

        "fields": ["temperature_2m"],

        "api_extra": {"temperature_unit": "fahrenheit"},

        "classify": classify_temp,

        "bottom_chart": "lines",

        "line_label": "Temperature (°F)",

        "label_extrema": True,  # print daily highs/lows on each model bar

        "colorbar": True,  # vertical scale on the right instead of a legend

        "legend": [],

        "legend_ncol": 1,

    },

}





# ---------------------------------------------------------------------------

# Prompts

# ---------------------------------------------------------------------------

def prompt_float(label):

    while True:

        try:

            return float(input(f"{label}: ").strip())

        except ValueError:

            print("  please enter a number")





def prompt_variable():

    print("\nAvailable variables:")

    for key, var in VARIABLES.items():

        print(f"  {key}. {var['name']}")

    while True:

        raw = input("\nSelect a variable (enter the number): ").strip()

        if raw in VARIABLES:

            return VARIABLES[raw]

        print("  please enter one of: " + ", ".join(VARIABLES))





def prompt_models():

    print("\nAvailable models:")

    for key, (name, _) in MODELS.items():

        print(f"  {key}. {name}")

    raw = input("\nSelect models (comma-separated numbers, or 'all'): ").strip().lower()

    if raw == "all":

        return list(MODELS.values())

    chosen = []

    for tok in raw.split(","):

        tok = tok.strip()

        if tok in MODELS:

            chosen.append(MODELS[tok])

    if not chosen:

        print("No valid models selected, defaulting to all.")

        return list(MODELS.values())

    return chosen





# ---------------------------------------------------------------------------

# Data fetching

# ---------------------------------------------------------------------------

def _clean(v):

    """Treat NaN as missing so it's shown as 'No Data' rather than 'no event'."""

    if v is None:

        return None

    try:

        return None if v != v else v

    except TypeError:

        return None





def fetch_one_model(lat, lon, model_id, var):

    """Request a single model. Raises on any failure (HTTP error, bad JSON...)."""

    url = "https://api.open-meteo.com/v1/forecast"

    params = {

        "latitude": lat,

        "longitude": lon,

        "hourly": ",".join(var["fields"]),

        "models": model_id,

        "forecast_days": 3,

        "timezone": "America/New_York",

    }

    params.update(var["api_extra"])

    resp = requests.get(url, params=params, timeout=30)

    try:

        body = resp.json()

    except ValueError:

        snippet = resp.text[:80].replace("\n", " ")

        raise RuntimeError(f"unreadable response (HTTP {resp.status_code}): {snippet!r}")

    if not resp.ok or (isinstance(body, dict) and body.get("error")):

        # Open-Meteo returns {"error": true, "reason": "..."} naming the problem

        reason = body.get("reason", resp.text) if isinstance(body, dict) else resp.text

        raise RuntimeError(f"HTTP {resp.status_code}: {reason}")

    if "hourly" not in body:

        raise RuntimeError("response had no hourly data")

    return body["hourly"]





def fetch_forecast(lat, lon, chosen_models, var):

    """Fetch each model on its own so a model that doesn't cover the location

    (or errors out) is simply shown as 'No Data' instead of killing the whole

    chart. Returns {"hourly": {...}} with fields suffixed by model id."""

    results = {}

    for name, model_id in chosen_models:

        try:

            results[model_id] = fetch_one_model(lat, lon, model_id, var)

        except (requests.RequestException, RuntimeError) as e:

            print(f"  {name}: no data ({e}) - will show as No Data")



    if not results:

        raise RuntimeError("none of the selected models returned data for this location")



    # use the first successful model's time axis as the reference and line

    # every other model up to it by timestamp

    ref_times = next(iter(results.values()))["time"]

    merged = {"time": ref_times}

    for model_id, hourly in results.items():

        lookup = {t: i for i, t in enumerate(hourly["time"])}

        for field in var["fields"]:

            values = hourly.get(field)

            if values is None:

                continue

            merged[f"{field}_{model_id}"] = [

                _clean(values[lookup[t]]) if t in lookup else None for t in ref_times

            ]

    return {"hourly": merged}





def build_series(data, model_id, fields, num_times):

    """Return one list per field for this model, handling the _<model_id>

    suffix Open-Meteo adds when multiple models are requested."""

    hourly = data["hourly"]

    series = []

    for field in fields:

        key = f"{field}_{model_id}"

        if key in hourly:

            series.append(hourly[key])

        else:

            # fall back to unsuffixed field if only one model was requested

            series.append(hourly.get(field, [None] * num_times))

    return series





# ---------------------------------------------------------------------------

# Support calculation for the bottom chart (support-style variables)

# ---------------------------------------------------------------------------

def compute_layer_support(var, chosen_models, model_series, num_hours):

    """For each hour, sum a weighted support score *separately* for each

    stacked layer. Each layer's total is normalized to 0-100% against only the

    models that actually have data that hour (models showing the gray "no

    data" hatch are excluded entirely, not counted as 0).



    Since every model falls in exactly one layer per hour, the layer values at

    a given hour add up to the overall support for that hour, which is what

    makes them stackable.



    Returns a list of numpy arrays, one per layer (same order as

    var["layers"]), each with one value per hour.

    """

    layer_keys = [key for key, _ in var["layers"]]

    out = {key: [] for key in layer_keys}



    for i in range(num_hours):

        totals = {key: 0.0 for key in layer_keys}

        available_models = 0  # models that actually have data this hour

        for _, model_id in chosen_models:

            values = [s[i] for s in model_series[model_id]]

            cls = var["classify"](*values)

            if cls == "nodata":

                continue  # doesn't count toward the total OR the denominator

            available_models += 1

            if cls is None:

                continue

            layer, _, weight = cls

            totals[layer] += weight



        max_possible = available_models * var["max_weight"]

        for key in layer_keys:

            out[key].append(totals[key] / max_possible * 100 if max_possible else 0)



    return [np.array(out[key], dtype=float) for key in layer_keys]





# ---------------------------------------------------------------------------

# Bottom chart A: stacked support chart (precip / wind / CAPE)

# ---------------------------------------------------------------------------

def draw_support_chart(ax2, var, chosen_models, model_series, times, smooth):

    layer_arrays = compute_layer_support(var, chosen_models, model_series, len(times))

    # cumulative boundary curves for stacking (row k = top edge of layer k)

    boundaries = np.cumsum(np.vstack(layer_arrays), axis=0)



    # plot each hourly value at the midpoint of its hour, matching the bars above

    point_times = [t + timedelta(minutes=30) for t in times]

    x_num = mdates.date2num(point_times)



    if smooth and len(x_num) >= 4:

        # cubic spline through every actual hourly boundary value - curves

        # the chart without altering or dropping any of the real numbers

        x_smooth = np.linspace(x_num.min(), x_num.max(), len(x_num) * 12)

        zero_line = np.zeros_like(x_smooth)

        plot_bounds = [

            np.clip(make_interp_spline(x_num, b, k=3)(x_smooth), 0, 100)

            for b in boundaries

        ]

        # enforce each boundary stays at/above the one beneath it, since a

        # spline can slightly overshoot near sharp jumps in the raw data

        for k in range(1, len(plot_bounds)):

            plot_bounds[k] = np.maximum(plot_bounds[k], plot_bounds[k - 1])

        x_plot = mdates.num2date(x_smooth)

    else:

        zero_line = np.zeros_like(x_num)

        plot_bounds = list(boundaries)

        x_plot = point_times



    lower = zero_line

    for k, (_, color) in enumerate(var["layers"]):

        ax2.fill_between(x_plot, lower, plot_bounds[k], color=color, linewidth=0)

        lower = plot_bounds[k]

    ax2.plot(x_plot, plot_bounds[-1], color="#333333", linewidth=1, alpha=0.6)



    # small dots mark the true hourly boundary values so no data is hidden by the curve

    for k, (_, color) in enumerate(var["layers"]):

        ax2.scatter(point_times, boundaries[k], color=color, s=10, zorder=5,

                    edgecolor="#333333", linewidth=0.4)



    ax2.set_ylim(0, 100)

    ax2.set_yticks([0, 25, 50, 75, 100])

    ax2.set_ylabel(var["support_label"], fontsize=9)





# ---------------------------------------------------------------------------

# Bottom chart B: one line per model + bold black average (temperature)

# ---------------------------------------------------------------------------

def _plot_line(ax, x_num, y, smooth, **kwargs):

    """Plot one line. Uses a cubic spline through the real hourly values when

    the series is complete; series with gaps are drawn straight so no values

    are invented across a hole."""

    ok = ~np.isnan(y)

    if not ok.any():

        return

    if smooth and ok.all() and len(x_num) >= 4:

        xs = np.linspace(x_num.min(), x_num.max(), len(x_num) * 12)

        ys = make_interp_spline(x_num, y, k=3)(xs)

        ax.plot(mdates.num2date(xs), ys, **kwargs)

    else:

        ax.plot(mdates.num2date(x_num), y, **kwargs)





def draw_line_chart(ax2, var, chosen_models, model_series, times, smooth):

    point_times = [t + timedelta(minutes=30) for t in times]

    x_num = mdates.date2num(point_times)



    cmap = plt.get_cmap("tab20")

    all_vals = []

    for n, (name, model_id) in enumerate(chosen_models):

        raw = model_series[model_id][0]  # first (only) field = the raw values

        vals = np.array([np.nan if v is None else v for v in raw], dtype=float)

        all_vals.append(vals)

        if np.isnan(vals).all():

            continue  # model has no data at all -> nothing to draw

        _plot_line(ax2, x_num, vals, smooth, color=cmap(n % 20),

                   linewidth=1.2, alpha=0.85, label=name, zorder=3)



    # average across only the models that have data at each hour

    stack = np.vstack(all_vals)

    counts = np.sum(~np.isnan(stack), axis=0)

    sums = np.nansum(stack, axis=0)

    avg = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    _plot_line(ax2, x_num, avg, smooth, color="black", linewidth=3.2,

               label="Model average", zorder=10)



    # freezing reference line

    ax2.axhline(32, color="#4a90d9", linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)



    # horizontal lines + labels every 5 F, with the 10s a bit darker so the

    # scale is easy to read at a glance

    lo, hi = ax2.get_ylim()

    lo = 5 * np.floor(lo / 5)

    hi = 5 * np.ceil(hi / 5)

    ax2.set_ylim(lo, hi)

    ax2.yaxis.set_major_locator(mticker.MultipleLocator(5))

    ax2.tick_params(axis="y", labelsize=8)

    for level in np.arange(lo, hi + 1, 5):

        if level % 10 == 0:

            ax2.axhline(level, color="#9a9a9a", linewidth=0.8, alpha=0.7, zorder=0)



    ax2.set_ylabel(var["line_label"], fontsize=9)

    # model-name legend sits below the chart (the right side is the color scale)

    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=6,

               frameon=False, fontsize=8)





# ---------------------------------------------------------------------------

# Daily high / low detection

# ---------------------------------------------------------------------------

def find_extrema(values, window=EXTREMA_WINDOW_HOURS, min_swing=EXTREMA_MIN_SWING_F):

    """Return indices of local highs and lows in a list of hourly values.



    A point counts when it is the max (or min) of everything within `window`

    hours on each side, and the window's total swing is at least `min_swing`.

    Needs a full window on both sides, so it is run on the full forecast and

    only then trimmed to the visible range. Ties pick the earliest hour.

    """

    vals = np.array([np.nan if v is None else v for v in values], dtype=float)

    found = []

    for i in range(window, len(vals) - window):

        if np.isnan(vals[i]):

            continue

        win = vals[i - window:i + window + 1]

        if np.isnan(win).all():

            continue

        if np.nanmax(win) - np.nanmin(win) < min_swing:

            continue

        if np.nanargmax(win) == window or np.nanargmin(win) == window:

            found.append(i)

    return found





# ---------------------------------------------------------------------------

# Vertical temperature color scale (right side of the figure)

# ---------------------------------------------------------------------------

def draw_temp_colorbar(fig, ax, ax2):

    """Tall vertical scale spanning both charts, built from TEMP_TIERS so it

    always matches the bar colors. One 5 F block per tier, labeled every 10 F."""

    pos_top = ax.get_position()

    pos_bot = ax2.get_position()

    x0 = pos_top.x1 + 0.025

    cax = fig.add_axes([x0, pos_bot.y0, 0.016, pos_top.y1 - pos_bot.y0])



    colors = [c for _, _, c, _ in TEMP_TIERS]

    cmap = mcolors.ListedColormap(colors)

    edges = [t for t, _, _, _ in TEMP_TIERS] + [TEMP_MAX + TEMP_STEP]

    norm = mcolors.BoundaryNorm(edges, cmap.N)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,

                      orientation="vertical", ticks=list(range(TEMP_MIN, TEMP_MAX + 1, 10)))



    labels = [str(v) for v in range(TEMP_MIN, TEMP_MAX + 1, 10)]

    labels[0] = f"≤{TEMP_MIN}"

    labels[-1] = f"{TEMP_MAX}+"

    cb.ax.set_yticklabels(labels, fontsize=9)

    cb.ax.tick_params(length=3, width=0.6)

    cb.outline.set_linewidth(0.6)

    cb.ax.set_title("°F", fontsize=10, fontweight="bold", pad=6)





# ---------------------------------------------------------------------------

# Plotting

# ---------------------------------------------------------------------------

def plot_timeline(lat, lon, data, chosen_models, var, forecast_hours=48, smooth=True):

    times = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]

    num_all = len(times)

    # data is requested in America/New_York, so compare against "now" in ET

    now = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)

    # trim to "now" -> now + forecast_hours

    idx_start = 0

    for i, t in enumerate(times):

        if t >= now - timedelta(hours=1):

            idx_start = i

            break

    idx_end = min(idx_start + forecast_hours, len(times))

    times = times[idx_start:idx_end]



    # precompute each model's trimmed series once, reused by both charts

    model_series = {}

    extrema = {}  # model_id -> visible-range indices of daily highs/lows

    for _, model_id in chosen_models:

        full = build_series(data, model_id, var["fields"], num_all)

        model_series[model_id] = [s[idx_start:idx_end] for s in full]

        if var.get("label_extrema"):

            # detect on the full forecast so highs/lows near the edges of the

            # visible window still have enough neighbors to be recognized

            extrema[model_id] = [i - idx_start for i in find_extrema(full[0])

                                 if idx_start <= i < idx_end]



    # left margin needs to scale with the longest model label so nothing gets clipped

    longest_label = max((len(name) for name, _ in chosen_models), default=10)

    left_margin = min(0.40, max(0.18, 0.011 * longest_label + 0.06))



    fig, (ax, ax2) = plt.subplots(

        2, 1, sharex=True,

        figsize=(14, 1.2 * len(chosen_models) + 4.5),

        gridspec_kw={"height_ratios": [len(chosen_models), 2.2], "hspace": 0.12},

    )



    # ---- top chart: one row of hourly bars per model ----

    for row, (name, model_id) in enumerate(chosen_models):

        series = model_series[model_id]

        y = len(chosen_models) - row - 1

        for i, t in enumerate(times):

            cls = var["classify"](*[s[i] for s in series])

            if cls is None:

                continue

            if cls == "nodata":

                ax.bar(t, height=0.8, bottom=y, width=timedelta(hours=1),

                       color=NODATA_COLOR, align="edge", edgecolor="white",

                       linewidth=0.2, hatch="///")

                continue

            _, color, _ = cls

            ax.bar(t, height=0.8, bottom=y, width=timedelta(hours=1),

                   color=color, align="edge", edgecolor="none")



        # daily high/low labels, e.g. "80°" centered on that hour's block

        for i in extrema.get(model_id, []):

            temp = series[0][i]

            if temp is None:

                continue

            r, g, b = mcolors.to_rgb(var["classify"](temp)[1])

            luminance = 0.299 * r + 0.587 * g + 0.114 * b

            ax.text(times[i] + timedelta(minutes=30), y + 0.4, f"{round(temp)}°",

                    ha="center", va="center", fontsize=8, fontweight="bold",

                    color="white" if luminance < 0.55 else "black", zorder=6)



    # y axis: model names

    ax.set_yticks([len(chosen_models) - i - 0.6 for i in range(len(chosen_models))])

    ax.set_yticklabels([name for name, _ in chosen_models], fontsize=11, fontweight="bold")

    ax.set_ylim(0, len(chosen_models))



    # x axis: time, in ET, two rows (time on top, bold date under each midnight)

    ax.set_xlim(times[0], times[-1] + timedelta(hours=1))



    tick_interval_hours = 3

    tick_times = []

    t = times[0].replace(minute=0, second=0, microsecond=0)

    while t.hour % tick_interval_hours != 0:

        t += timedelta(hours=1)

    while t <= times[-1] + timedelta(hours=1):

        tick_times.append(t)

        t += timedelta(hours=tick_interval_hours)



    ax.set_xticks(tick_times)

    ax.grid(axis="x", color="lightgray", linewidth=0.5)

    for spine in ("top", "right", "left"):

        ax.spines[spine].set_visible(False)

    # x tick labels live on the bottom axes, since the two charts share the x-axis

    plt.setp(ax.get_xticklabels(), visible=False)



    ax.set_title(f"Model {var['title']} Forecast — lat {lat}, lon {lon}\n"

                 f"Generated {now.strftime('%Y-%m-%d %H:%M')} ET", fontsize=13)



    # ---- bottom chart: support chart OR model line chart ----

    if var.get("bottom_chart", "support") == "lines":

        draw_line_chart(ax2, var, chosen_models, model_series, times, smooth)

    else:

        draw_support_chart(ax2, var, chosen_models, model_series, times, smooth)



    ax2.grid(axis="both", color="lightgray", linewidth=0.5)

    for spine in ("top", "right"):

        ax2.spines[spine].set_visible(False)



    time_labels = [tt.strftime("%I%p").lstrip("0") for tt in tick_times]

    ax2.set_xticklabels(time_labels, fontsize=9)

    for lbl, tt in zip(ax2.get_xticklabels(), tick_times):

        if tt.hour == 0:

            lbl.set_fontweight("bold")



    # bold date row beneath each midnight tick (data coords for x, axes-fraction for y)

    for tt in tick_times:

        if tt.hour == 0:

            ax2.text(tt, -0.38, tt.strftime("%b %d"), transform=ax2.get_xaxis_transform(),

                     ha="center", va="top", fontsize=10, fontweight="bold")



    use_colorbar = var.get("colorbar", False)

    if not use_colorbar:

        # color legend placed below both charts

        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in var["legend"]]

        labels = [l for l, _ in var["legend"]]

        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.12),

                   ncol=var["legend_ncol"], frameon=False, fontsize=9)



    fig.subplots_adjust(left=left_margin, right=0.91 if use_colorbar else 0.97,

                        top=0.90, bottom=0.22)



    if use_colorbar:

        draw_temp_colorbar(fig, ax, ax2)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    suffix = "spline" if smooth else "nospline"
    out_path = os.path.join(
        out_dir,
        f"{var['slug']}_forecast_chart_{suffix}.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to: {out_path}")
    return fig

def main():
    print("=== Philadelphia Multi-Model Forecast ===")

    # Philadelphia city-center coordinates
    lat = 39.9526
    lon = -75.1652

    # Automatically use every model in the catalog
    chosen_models = list(MODELS.values())

    # Generate one PNG for each configured variable
    for var in VARIABLES.values():
        print(f"\nGenerating {var['name']} forecast...")

        try:
            data = fetch_forecast(lat, lon, chosen_models, var)
            plot_timeline(
                lat,
                lon,
                data,
                chosen_models,
                var,
                smooth=True
            )
        except Exception as e:
            print(f"Failed to generate {var['name']}: {e}")

    print("\nForecast generation finished.")


if __name__ == "__main__":
    main()
