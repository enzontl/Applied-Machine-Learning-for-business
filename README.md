# Data

Two datasets from [SNCF Open Data](https://ressources.data.sncf.com), licensed under [ODbL](https://opendatacommons.org/licenses/odbl/).

---

## Files

### `regularite-mensuelle-tgv-aqst.csv`
Monthly TGV punctuality by route, from January 2018 to present.  
11,834 rows × 26 columns. Separator: `;`

**Download:**
```
https://ressources.data.sncf.com/explore/dataset/regularite-mensuelle-tgv-aqst/download/?format=csv&use_labels_for_header=true
```

**Columns:**

| Column | Type | Description |
|---|---|---|
| `Date` | date | Year-month (YYYY-MM) |
| `Service` | str | `National` or `International` |
| `Gare de départ` | str | Departure station |
| `Gare d'arrivée` | str | Arrival station |
| `Durée moyenne du trajet` | int | Average journey time (min) |
| `Nombre de circulations prévues` | int | Scheduled trains |
| `Nombre de trains annulés` | int | Cancelled trains |
| `Nombre de trains en retard au départ` | int | Trains delayed at departure |
| `Retard moyen des trains en retard au départ` | float | Avg delay at departure — delayed trains only (min) |
| `Retard moyen de tous les trains au départ` | float | Avg delay at departure — all trains (min) |
| `Nombre de trains en retard à l'arrivée` | int | Trains delayed at arrival |
| `Retard moyen des trains en retard à l'arrivée` | float | Avg delay at arrival — delayed trains only (min) |
| `Retard moyen de tous les trains à l'arrivée` | float | Avg delay at arrival — all trains (min) |
| `Nombre trains en retard > 15min` | int | Trains delayed > 15 min |
| `Nombre trains en retard > 30min` | int | Trains delayed > 30 min |
| `Nombre trains en retard > 60min` | int | Trains delayed > 60 min |
| `Prct retard pour causes externes` | float | % delays — external causes (weather, passengers) |
| `Prct retard pour cause infrastructure` | float | % delays — infrastructure |
| `Prct retard pour cause gestion trafic` | float | % delays — traffic management |
| `Prct retard pour cause matériel roulant` | float | % delays — rolling stock |
| `Prct retard pour cause gestion en gare et réutilisation de matériel` | float | % delays — station management |
| `Prct retard pour cause prise en compte voyageurs (affluence, gestions PSH, correspondances)` | float | % delays — passenger handling |
| `Commentaire annulations` | str | Free-text notes — 100% null, unused |
| `Commentaire retards au départ` | str | Free-text notes — 100% null, unused |
| `Commentaire retards à l'arrivée` | str | Free-text notes — mostly null |
| `Retard moyen trains en retard > 15 (si liaison concurrencée par vol)` | float | Avg delay > 15 min (air-competing routes only) |

**Notes:**
- The six `Prct retard pour cause ...` columns sum to ~100% on rows with delays
- 200 rows have zero delayed trains (excluded from modelling)
- 78 rows have delayed trains but all cause columns = 0 (data quality issue, excluded)

---

### `gares-de-voyageurs.csv`
Reference table of SNCF passenger stations. 2,782 rows × 7 columns. Separator: `;`

**Download:**
```
https://ressources.data.sncf.com/explore/dataset/gares-de-voyageurs/download/?format=csv&use_labels_for_header=true
```

**Columns:**

| Column | Type | Description |
|---|---|---|
| `Nom` | str | Station name |
| `Trigramme` | str | 3-letter station code |
| `Segment(s) DRG` | str | Station category — `A` (major), `B` (medium), `C` (small) |
| `Position géographique` | str | GPS coordinates as `"lat,lon"` string — parse with `str.split(',')` |
| `Code commune` | int | INSEE municipality code |
| `Code(s) UIC` | int | UIC station code |
| `id` | str | UUID |

**Notes:**
- GPS coordinates are stored as a single `"lat,lon"` string — split before use
- `Segment(s) DRG` can contain multiple values separated by `;` (e.g. `A;B`)
- Join key with the TGV dataset: `Nom` ↔ `Gare de départ` / `Gare d'arrivée`  
  ⚠ Partial match only (~26/60 departure stations match directly) — name normalisation required

---

## Usage

```python
import pandas as pd

punctuality = pd.read_csv(
    "raw/regularite-mensuelle-tgv-aqst.csv",
    sep=";",
    parse_dates=["Date"]
)

stations = pd.read_csv("raw/gares-de-voyageurs.csv", sep=";")

# Parse GPS coordinates
stations[["lat", "lon"]] = (
    stations["Position géographique"]
    .str.split(",", expand=True)
    .astype(float)
)
```

---

## License

Source: SNCF Open Data — © SNCF  
License: [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/)  
Downloaded: 2026-03