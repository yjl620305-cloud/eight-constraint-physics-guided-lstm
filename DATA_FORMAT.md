# Local data format (data are not distributed)

The script expects a local folder with this structure:

```text
<DATA_ROOT>/
├── 察汗乌苏数据/
│   └── data/
│       ├── Ts01.csv
│       └── MD03-C2.csv
└── 监测指标公式及示例测点参数统计/
    └── 库水位.xlsx
```

## `Ts01.csv` and `MD03-C2.csv`

Each CSV must include:

- `time`: parseable date/time.
- `data`: numeric measurement.

`Ts01.csv` is read as temperature; `MD03-C2.csv` is read as the displacement target. The script does not publish or generate these files.

## `库水位.xlsx`

The validated loader expects the timestamp in Excel column B and water level in column G (zero-based column indices 1 and 6), matching the authorized experiment workbook.

## Alignment and units

- Temperature values are assumed to be in degrees Celsius.
- Displacement values are assumed to be in millimetres.
- Water-level units must match the manuscript and should be checked by the authors. The default minimum informative water-level change is 0.01 in the same units as the workbook.
- Water-level observations are aligned backward to the temperature timestamp with a maximum tolerance of three days. Displacement observations are matched to the nearest timestamp within the same tolerance.
- Rows without aligned target or environmental measurements are dropped; environmental values are forward-filled only after alignment. Missing displacement labels are not interpolated.
- The target displacement, timestamp, and input features are robust-scaled using the training interval only. The scaling parameters are exported to `scalers.json`.

The above conventions correspond to the authors' validated experiment and the final manuscript.

