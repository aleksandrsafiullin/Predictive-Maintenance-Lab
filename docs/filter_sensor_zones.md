# Filter sensor zones

Filter zones describe measured pressure bands for each HSE observation. They do
not represent remaining life, time before an event, or a maintenance deadline. Policy
`filter_sensor_zones_v2` uses provisional yellow boundary `300 Pa`, identified as
`filter_pressure_warning_300pa_v1`, pending expert validation. Every runtime row
records this policy, its provenance, measured pressure, flow, and dust feed.

| Color | Sensor zone | Rule | Meaning |
| --- | --- | --- | --- |
| Green | `below_provisional_pressure_band` | Usable pressure below 300 Pa, positive finite flow, and finite dust feed | Below the provisional warning band only; this does not mean healthy |
| Yellow | `pressure_warning_band` | Usable pressure from 300 Pa to below 600 Pa, positive finite flow, and finite dust feed | Provisional absolute pressure warning band |
| Red | `configured_600_pa_limit` | Usable measured pressure at or above 600 Pa | Configured monitoring hard limit |
| Gray | `unknown` | Pressure is missing/invalid or unusable; or below 600 Pa with missing/nonpositive flow or missing feed | Required measurement context is unavailable |

The separate HSE event convention remains the first differential-pressure
measurement strictly **above** 600 Pa. The displayed sensor zone follows the
monitoring hard-limit convention **at or above** 600 Pa, so exactly 600 Pa is red
in the UI. No time value, event label, RUL value, or future observation determines
the displayed color.

## Supplementary measured evidence

When a provisional train reference matches the current recorded dust, flow, and
feed regime, runtime reports pressure residuals relative to its median and robust
scale. It can also report a trend over the last five measurements when flow/feed
bins remain unchanged. The trend diagnostic requires a pressure rise greater than
three reference robust scales and at least three of four rising steps. These fields
are supplementary evidence and reason codes only; they never change the color below
300 Pa. When the reference is absent or unmatched, the absolute pressure bands still
apply and the result does not claim a confirmed healthy state.

`zones-labels --dataset filters` writes one row for every admitted measurement,
including retained attention rows, to a versioned `labels.csv` plus a manifest under
`runs/_zones/filters/label_artifacts/`. The manifest binds dataset version and
fingerprint, zone policy, row count, class counts by train/validation/test split, and
the CSV SHA-256. Labels use only measured pressure, row quality, flow, and dust feed;
endpoint and RUL fields are not read. The export classification and runtime display
use the same pressure-band and context gates.

The flattened `state_history.parquet` from monitoring evaluation includes the zone
name, color, status, reason, supplementary evidence, pressure, flow/feed values, and
policy version. Old saved evaluations without a `sensor_zone` row are identified in
the UI as predating this feature and need a new evaluation. A zone-label CSV treats
each accepted row as a contemporaneous observation; runtime checks can return gray
when that observation is stale relative to a later `as_of` time.

Flow/feed values are shown as recorded source values. HSE Figure 6 labels the
source time axis `Time / s`; the CSV advances in 0.1-second steps. Trends and
replay timestamps therefore use source seconds. The RUL column's unit is
inferred from its relation to Time, rather than stated separately in the source
schema. This remains a laboratory condition view, not an operational safety
claim; see [data units and endpoints](data_units_and_endpoints.md).
