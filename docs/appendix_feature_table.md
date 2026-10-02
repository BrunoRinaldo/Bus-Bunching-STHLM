# Appendix A. Model features

The tabular models (logistic regression and HistGradientBoosting) use the 37
features below (`ALL_COLS` in `src/models.py`). All features are computed at
stop $n$ or earlier. The only look-ahead column is the target.

## A.1 Notation

| Symbol | Meaning |
|---|---|
| $i$, $n$ | trip $i$ at stop sequence $n$ |
| $\ell$ | leader of trip $i$: the trip scheduled just before $i$ at the same stop, line and direction on the same day |
| $t^{a}_{i,n},\ t^{d}_{i,n}$ | observed arrival / departure time (s) |
| $s^{a}_{i,n}$ | scheduled arrival time (s) |
| $x^{\mathrm{RT}}_{i,n}$ | value taken as-is from the SL GTFS-RT feed |
| $x^{\mathrm{GTFS}}_{p}$ | value taken as-is from static GTFS for stop $p$ |
| $x^{\mathrm{OM}}_{g,h}$ | value taken as-is from the Open-Meteo archive at grid point $g$ and UTC hour $h$ |
| $g(p)$ | 0.1° grid point nearest to stop $p$ |
| $h(t)$ | UTC hour of local time $t$ |
| $\mathbb{1}[\cdot]$ | indicator, 1 if the condition holds, else 0 |
| $n_0,\ n_1$ | first and last observed stop sequence of trip $i$ |

Imported (external) values are written $x_{i,n} = x^{\mathrm{source}}$: the
feature is the source column, unchanged.

## A.2 Feature table

### Imported from GTFS-RT and static GTFS

| Feature | Formula | Description |
|---|---|---|
| `observed_arrival_delay` | $\delta^{a}_{i,n} = \delta^{a,\mathrm{RT}}_{i,n}$ | Arrival delay against the timetable (s), as reported by the feed. |
| `observed_departure_delay` | $\delta^{d}_{i,n} = \delta^{d,\mathrm{RT}}_{i,n}$ | Departure delay against the timetable (s), as reported by the feed. |
| `stop_sequence` | $n = n^{\mathrm{RT}}_{i}$ | Position of the stop within the trip pattern. |
| `dist_traveled` | $D_{i,n} = D^{\mathrm{RT}}_{i,n}$ | Length of the link from the previous stop to stop $n$ (m). |
| `LineNumber` | $L_i = L^{\mathrm{RT}}_{i}$ | Bus line (categorical, one-hot for logistic regression). |
| `direction_id` | $r_i = r^{\mathrm{RT}}_{i}$ | Direction of travel (categorical). |
| `has_parent_station` | $\mathbb{1}[\mathrm{parent\_station}^{\mathrm{GTFS}}_{p} \neq \varnothing]$ | Stop belongs to a larger stop complex. |

### Imported from Open-Meteo

| Feature | Formula | Description |
|---|---|---|
| `temperature_2m` | $T_{i,n} = T^{\mathrm{OM}}_{g(p),\,h(t^{a}_{i,n})}$ | Air temperature at 2 m (°C). |
| `precipitation` | $P_{i,n} = P^{\mathrm{OM}}_{g(p),\,h(t^{a}_{i,n})}$ | Precipitation in the hour (mm). |
| `wind_speed_10m` | $W_{i,n} = W^{\mathrm{OM}}_{g(p),\,h(t^{a}_{i,n})}$ | Wind speed at 10 m (m/s). |

### Current state at stop n

| Feature | Formula | Description |
|---|---|---|
| `headway_ratio` | $H_{i,n} = \dfrac{t^{a}_{i,n} - t^{a}_{\ell,n}}{s^{a}_{i,n} - s^{a}_{\ell,n}}$ | Observed over scheduled headway to the leader. 1 = on plan, 0 = buses together, negative = overtaken. Null if the observed headway is outside $[-1800, 10800]$ s or the scheduled headway is $\le 0$. |
| `dwell` | $d_{i,n} = t^{d}_{i,n} - t^{a}_{i,n}$ | Dwell time (s). Null at the first and last stop and when negative. |
| `dwell_rel_median` | $d_{i,n} - \operatorname{median}_{j}\, d_{j,p}$ | Dwell relative to the yearly median dwell at the same stop $p$. |
| `speed_mps` | $v_{i,n} = \dfrac{D_{i,n}}{t^{a}_{i,n} - t^{d}_{i,n-1}}$ | Link speed into stop $n$ (m/s). Null if run time $\le 0$. |

### Trajectory of the same trip

| Feature | Formula | Description |
|---|---|---|
| `headway_ratio_lag1/2/3` | $H_{i,n-m},\ m \in \{1,2,3\}$ | Headway ratio at the three previous stops. |
| `closing_speed_3` | $\dfrac{H_{i,n} - H_{i,n-3}}{3}$ | Average change in headway ratio per stop over the last three stops. Negative = gap closing. |
| `cum_delay_growth_3` | $\sum_{m=0}^{2} \left(\delta^{d}_{i,n-m} - \delta^{a}_{i,n-m}\right)$ | Delay added at stops $n-2$ to $n$ (s). |
| `speed_lag1/2` | $v_{i,n-m},\ m \in \{1,2\}$ | Link speed on the two previous links. |

### Leader

| Feature | Formula | Description |
|---|---|---|
| `leader_dwell` | $d_{\ell,n}$ | Leader's dwell at stop $n$. |
| `leader_arrival_delay` | $\delta^{a}_{\ell,n}$ | Leader's arrival delay at stop $n$. |
| `leader_delay_growth` | $\delta^{d}_{\ell,n} - \delta^{a}_{\ell,n}$ | Delay the leader added at stop $n$. |

### Position and time

| Feature | Formula | Description |
|---|---|---|
| `route_frac` | $\dfrac{n - n_0}{n_1 - n_0}$ | Share of the trip completed (0 = start, 1 = end). |
| `hour_of_day` | $\operatorname{hour}(s^{a}_{i,n})$ | Hour of the scheduled arrival. |
| `day_of_week` | $\operatorname{dow}(s^{a}_{i,n})$ | Day of week of the scheduled arrival (0 = Sunday). |
| `month_of_year` | $\operatorname{month}(s^{a}_{i,n})$ | Month of the scheduled arrival. |

### Stop attributes and disruption

| Feature | Formula | Description |
|---|---|---|
| `is_interchange` | $\mathbb{1}[p \in \mathrm{transfers}^{\mathrm{GTFS}}]$ | Stop appears as origin or destination in `transfers.csv`. |
| `n_routes_serving` | $\#\{L : L \text{ observed at } p \text{ in January}\}$ | Number of distinct bus lines serving the stop (1 if unknown). |
| `under_active_alert` | $\mathbb{1}[\exists a:\ p_a = p,\ \tau^{\mathrm{start}}_a \le t^{a}_{i,n} \le \tau^{\mathrm{end}}_a]$ | A service alert is active for the stop at arrival time. |

### Derived weather

| Feature | Formula | Description |
|---|---|---|
| `temperature_2m_sq` | $T_{i,n}^{2}$ | Lets logistic regression fit a U-shaped temperature effect. |
| `precip_3h` | $\sum_{m=0}^{2} P^{\mathrm{OM}}_{g(p),\,h-m}$ | Precipitation over the last three hours (mm). |
| `is_precip` | $\mathbb{1}[P_{i,n} > 0.1]$ | Rain on/off. |
| `is_snowing` | $\mathbb{1}[S^{\mathrm{OM}}_{g(p),h} > 0]$ | Snowfall in the hour. |
| `ice_risk` | $\mathbb{1}[T_{i,n} < 1 \wedge P_{i,n} > 0]$ | Slippery conditions. |
| `wx_group` | $f\!\left(\mathrm{WMO}^{\mathrm{OM}}_{g(p),h}\right)$ | WMO weather code mapped to clear, cloudy, fog, drizzle, rain, snow, rain showers, snow showers, thunderstorm or other. |

## A.3 Target

| Column | Formula | Description |
|---|---|---|
| `y_k{k}` | $y^{(k)}_{i,n} = \mathbb{1}\!\left[\,\lvert H_{i,n+k} \rvert < 0.25\,\right]$ | Bunched $k \in \{1,2,3,5,8\}$ stops ahead. Null if the trip ends before $n+k$ or $H_{i,n+k}$ is null. |

The sequence model uses $H$, $d$ and $\delta^{d} - \delta^{a}$ at stops
$n-5, \dots, n$ as input, with the same target.
