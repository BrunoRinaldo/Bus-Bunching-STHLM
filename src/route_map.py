"""Figure 23: the 8 bus lines and the 9 weather grid cells on a satellite basemap.

Line geometry: there is no shapes file in data/raw, so each line is drawn by
joining its stops in stop_sequence order along its most frequent shape_id per
direction (straight segments between stops, not the street geometry).

Weather cells: weather.py rounds each stop to 0.1 degree, so grid point
(lat, lon) covers lat +/- 0.05, lon +/- 0.05.

Basemap: Esri World Imagery tiles, downloaded once into data/cache/esri_tiles/.
"""
import io
import math
import os

import duckdb
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

from style import INK2

BUS = "data/processed/bus_subset.parquet"
WEATHER = "data/processed/weather.parquet"
STOPS = "data/raw/stops.csv"
TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TILE_DIR = "data/cache/esri_tiles"
ZOOM = 13
CELL = 0.1

# categorical slots 1-8 (dark-surface steps: the imagery is dark), lines in numeric order
COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]


def merc(lon, lat):
    """lon/lat in degrees -> Web Mercator metres."""
    r = 6378137.0
    x = np.radians(lon) * r
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * r
    return x, y


def tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def tile_bounds(tx, ty, z):
    """Web Mercator extent of tile (tx, ty)."""
    n, half = 2 ** z, 20037508.342789244
    size = 2 * half / n
    return -half + tx * size, half - (ty + 1) * size, size


def get_tile(z, x, y):
    path = f"{TILE_DIR}/{z}/{x}/{y}.jpg"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(TILE_URL.format(z=z, x=x, y=y), timeout=30,
                         headers={"User-Agent": "bus-bunching-thesis-figure"})
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return np.asarray(Image.open(path).convert("RGB"))


def basemap(lon0, lat0, lon1, lat1, z):
    """Stitch the tiles covering the box; returns image and its Mercator extent."""
    x0, y0 = tile_xy(lon0, lat1, z)
    x1, y1 = tile_xy(lon1, lat0, z)
    xs, ys = range(int(x0), int(x1) + 1), range(int(y0), int(y1) + 1)
    rows = [np.hstack([get_tile(z, x, y) for x in xs]) for y in ys]
    img = np.vstack(rows)
    left, _, size = tile_bounds(xs[0], ys[0], z)
    top = tile_bounds(xs[0], ys[0], z)[1] + size
    right, bottom = left + size * len(xs), top - size * len(ys)
    return img, (left, right, bottom, top)


def line_paths(con):
    """Ordered stop coordinates per (line, direction) along the most frequent shape."""
    return con.execute(f"""
        WITH top AS (
            SELECT LineNumber, direction_id, shape_id, count(*) n,
                   row_number() OVER (PARTITION BY LineNumber, direction_id ORDER BY count(*) DESC) rk
            FROM '{BUS}' GROUP BY 1, 2, 3
        ),
        seq AS (
            SELECT b.LineNumber, b.direction_id, b.observed_stop_id, median(b.stop_sequence) s
            FROM '{BUS}' b JOIN top t USING (LineNumber, direction_id, shape_id)
            WHERE t.rk = 1 GROUP BY 1, 2, 3
        )
        SELECT seq.*, st.stop_lat, st.stop_lon
        FROM seq JOIN read_csv_auto('{STOPS}', delim=';') st ON seq.observed_stop_id = st.stop_id
        ORDER BY LineNumber, direction_id, s
    """).df()


def main():
    con = duckdb.connect()
    paths = line_paths(con)
    grid = con.execute(f"SELECT DISTINCT grid_lat, grid_lon FROM '{WEATHER}' ORDER BY 1, 2").df()
    lines = sorted(paths.LineNumber.unique(), key=int)

    lon0, lon1 = grid.grid_lon.min() - CELL / 2 - 0.03, grid.grid_lon.max() + CELL / 2 + 0.03
    lat0, lat1 = grid.grid_lat.min() - CELL / 2 - 0.015, grid.grid_lat.max() + CELL / 2 + 0.015
    img, ext = basemap(lon0, lat0, lon1, lat1, ZOOM)
    (bx0, bx1), (by0, by1) = merc(np.array([lon0, lon1]), np.array([lat0, lat1]))

    # explicit layout (inches): header, map at its true aspect, legend column, footer
    map_w, legend_w, side, head, foot = 8.6, 1.9, 0.25, 0.95, 0.35
    map_h = map_w * (by1 - by0) / (bx1 - bx0)
    fig_w, fig_h = side + map_w + legend_w, head + map_h + foot
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([side / fig_w, foot / fig_h, map_w / fig_w, map_h / fig_h])
    ax.imshow(img, extent=ext, interpolation="lanczos", zorder=0)
    ax.imshow(np.zeros((1, 1, 4)) + [0, 0, 0, 0.25], extent=ext, zorder=1)  # dim imagery so lines read

    halo = [pe.Stroke(linewidth=4.2, foreground="white", alpha=0.9), pe.Normal()]
    pts = np.column_stack(merc(paths.stop_lon.values, paths.stop_lat.values))
    line_of = paths.LineNumber.values

    def clearance(xy, exclude=None):
        """Distance from a point to the nearest stop of any other line (or any line)."""
        m = line_of != exclude if exclude is not None else slice(None)
        return np.min(np.hypot(*(pts[m] - xy).T))

    pad = 0.012  # degrees, label inset from the cell corner
    for _, g in grid.iterrows():
        la, lo = g.grid_lat, g.grid_lon
        cl = np.array([lo - CELL / 2, lo + CELL / 2, lo + CELL / 2, lo - CELL / 2, lo - CELL / 2])
        ca = np.array([la - CELL / 2, la - CELL / 2, la + CELL / 2, la + CELL / 2, la - CELL / 2])
        x, y = merc(cl, ca)
        ax.fill(x, y, facecolor="white", alpha=0.08, zorder=2)
        ax.plot(x, y, color="white", linewidth=1.3, linestyle=(0, (5, 3)), zorder=3)
        # label in whichever corner is furthest from the bus lines; probe the label's centre
        corners = [(sx, sy) for sx in (-1, 1) for sy in (-1, 1)]
        probe = lambda c: np.array(merc(lo + c[0] * (CELL / 2 - 0.03), la + c[1] * (CELL / 2 - pad)))
        sx, sy = max(corners, key=lambda c: clearance(probe(c)))
        tx, ty = merc(lo + sx * (CELL / 2 - 0.004), la + sy * (CELL / 2 - 0.004))
        ax.text(tx, ty, f"{la:.1f}°N {lo:.1f}°E", fontsize=7.5, color="white",
                ha="right" if sx > 0 else "left", va="top" if sy > 0 else "bottom", zorder=6,
                path_effects=[pe.withStroke(linewidth=2, foreground="black", alpha=0.6)])

    placed = []
    for line, col in zip(lines, COLORS):
        p = paths[paths.LineNumber == line]
        for _, d in p.groupby("direction_id"):
            x, y = merc(d.stop_lon.values, d.stop_lat.values)
            ax.plot(x, y, color=col, linewidth=2.2, solid_capstyle="round", zorder=4, path_effects=halo)
        # label on the stop furthest from other lines and from labels already placed
        cand = np.column_stack(merc(p.stop_lon.values, p.stop_lat.values))
        score = [min([clearance(c, exclude=line)] + [np.hypot(*(c - q)) for q in placed]) for c in cand]
        best = cand[int(np.argmax(score))]
        placed.append(best)
        ax.text(*best, line, fontsize=8, fontweight="bold", color="#0b0b0b", ha="center", va="center",
                zorder=7, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=1.6))

    ax.set_xlim(bx0, bx1)
    ax.set_ylim(by0, by1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Line2D([], [], color=c, linewidth=3, label=f"line {l}") for l, c in zip(lines, COLORS)]
    handles.append(plt.Line2D([], [], color="#52514e", linewidth=1.3, linestyle=(0, (5, 3)),
                              label="weather cell (0.1°)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)

    n_stops = con.execute(f"SELECT count(DISTINCT observed_stop_id) FROM '{BUS}'").fetchone()[0]
    fig.suptitle("Study area: the 8 bus lines and the 9 Open-Meteo weather cells, Stockholm",
                 fontsize=13, x=side / fig_w, y=1 - 0.12 / fig_h, ha="left", va="top")
    fig.text(side / fig_w, 1 - 0.45 / fig_h, f"{n_stops} stops; lines drawn along each line's main route variant, joined stop to stop "
             "(straight segments, not street geometry);\neach stop takes weather from the cell it falls in",
             fontsize=9, color=INK2, ha="left", va="top")
    fig.text(side / fig_w, 0.1 / fig_h, "Imagery: Esri World Imagery (Esri, Maxar, Earthstar Geographics, GIS User Community)",
             fontsize=7, color="#898781", ha="left")
    fig.savefig("figures/fig23_study_area_map.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
