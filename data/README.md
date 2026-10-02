# Data

The dataset is not part of this repository. It is available from the corresponding author of
the paper on reasonable request.

| | |
|---|---|
| Source | DTU Database on Wind Characteristics: one wind turbine, 10-minute averaged measurements |
| File | `data/total_dataset.csv` (place it here and do not modify it) |
| Format | `;` separator, one header row, 4464 rows, 5 float64 columns, no timestamps |
| SHA-256 | `5bc783670f61835114c4a0fecef90c075c1041b85a97ac294f8da5316a876ebc` |

| Column | Unit | Role |
|---|---|---|
| Temperature | °C | feature (qubit q0) |
| Pressure | hPa | feature (qubit q1) |
| Theta | ° (wind direction) | feature (qubit q2) |
| Velocity | m/s (wind speed) | feature (qubit q3) |
| Power | kW | target |

`python scripts/verify_provenance.py` and `python scripts/inspect_data.py` check the file. The
checksum is also verified whenever the file is loaded.

Without the dataset, `scripts/reproduce_paper.py` still regenerates every table and figure
except T1, T1b, and A1 (see the main README). Training and reruns need the dataset.
