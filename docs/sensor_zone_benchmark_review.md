# Sensor-zone benchmark review

## Evidence and scope

These are laboratory comparisons of **measured sensor states**, not verified
industrial health labels. The [XJTU-SY dataset authors](https://biaowang.tech/xjtu-sy-bearing-datasets/)
describe 15 complete bearing trajectories, with a 1.28-second vibration sample
recorded once per minute. The bundled HSE source document,
`data/raw/filters/Preventive to Predictive Maintenance dataset.pdf`, labels the
horizontal axis of Figure 6 (p. 5) `Time / s`; p. 7 says `Sampling` is in Hz,
and p. 6 defines the filter event when differential pressure **exceeds** 600 Pa.
In the supplied test CSV, `Time` advances by 0.1 and `Time + RUL` is constant
within each unit, so the two columns share the same duration scale. The HSE
[source paper](https://papers.phmsociety.org/index.php/ijphm/article/download/3087/1835)
describes the differential-pressure test bench and right-censored life tests.

The HSE document does **not** define a 300 Pa warning boundary. The
[ISO 13373-1 overview](https://www.iso.org/standard/21831.html) gives general
vibration-monitoring procedures, while the [ISO 20816-1 overview](https://committee.iso.org/cms/live/live/en/sites/isoorg/contents/data/standard/06/31/63180.html)
says evaluation criteria depend on machine class and measurement method. Neither
provides a directly transferable threshold for our XJTU-SY accelerometer RMS
ratio. We therefore compare policies within the supplied data and keep their
engineering assumptions visible.

## Is 0.790 validation / 0.956 test a high score?

For reproducing the **existing signal rule**, the test value is strong. The
metric is the average over bearings of each bearing's average recall across
the classes present on that bearing. It is not ordinary row accuracy. The run
`zones_20260928_200845_2f9991` has these results:

| Comparison | Validation: 3 bearings | Test: 3 bearings |
| --- | ---: | ---: |
| Always predict green | 0.444 | 0.389 |
| Trained GRU, unit-balanced balanced accuracy | 0.790 | 0.956 |
| Direct sensor rule that generated the weak labels | 1.000 | 1.000 |

On test, the GRU recalled 203/204 green, 18/23 yellow and 271/278 red rows.
On validation it missed the **only red row** on `Bearing1_4`; this single
bearing scored 0.500 and pulled the three-bearing mean down to 0.790. The
validation/test difference is therefore sensitive to the small, uneven set of
independent trajectories. Neighboring one-minute rows share most of their
20-measurement input history and must not be counted as independent bearings.

The labels were generated from the same present and past vibration measurements
used by the model. The direct rule scores 1.000 by construction. Thus 0.956 is
**good rule agreement**, but it does not show that the thresholds detect a true
fault, that the model predicts a *future* entry into red, or that the GRU is
better than the direct rule for showing the *current* zone. We should use the
transparent sensor rule for current-state display. A future-red forecast needs
a separate horizon-based target and evaluation.

## Filter warning-boundary sensitivity

The table uses the **admitted** 39 training and 10 validation filter units;
`Train_28` is excluded. Four training units and one validation unit have an
observed first pressure sample strictly above 600 Pa; the other 35 and 9 units
are right-censored. We used the original CSV `Time` in seconds for lead times,
with usable pressure, positive flow and finite dust-feed context. A threshold
is reached at the first usable pressure measurement at or above that value.
No threshold was chosen using the test split.

| Yellow boundary | Training events reached; median lead to >600 Pa | Censored training units crossing | Validation events reached; lead | Censored validation units crossing |
| ---: | ---: | ---: | ---: | ---: |
| 200 Pa | 4/4; 28.6 s | 22/35 | 1/1; 24.9 s | 3/9 |
| **300 Pa (current)** | **4/4; 21.0 s** | **12/35** | **1/1; 15.8 s** | **1/9** |
| 400 Pa | 4/4; 13.7 s | 2/35 | 1/1; 10.4 s | 0/9 |
| 500 Pa | 4/4; 6.55 s | 1/35 | 1/1; 5.5 s | 0/9 |

Earlier thresholds gave more lead and more crossings in censored units.
**Crossing in a censored unit is not a confirmed false alarm**: its later
outcome was not observed. At the 600 Pa red boundary, the five observed events
are reached at the event sample itself, giving zero predictive lead; reaching
the event-defining pressure is a current-state observation, not a forecast.
The seconds here are accelerated laboratory time, not a useful maintenance
response interval established for real equipment.

For the lab, retaining **300 Pa as a provisional yellow band** is a transparent
middle choice. The five observed event trajectories support that it precedes
the experimental limit in this small cohort; they do not establish it as an
optimal alarm threshold. A 400 Pa boundary would reduce censored crossings in
these splits but shorten the observed lead. The preferred tradeoff requires a
specified cost of extra inspections versus delayed warnings. Green below 300 Pa
means only "below this provisional pressure band," not confirmed healthy.

## Next evaluation target

The present bearing GRU classifies the current zone. The older RUL models
estimated remaining time and are archived from active runs. To forecast **entry
into red**, define a fixed future horizon on train data, handle right-censored
filter prefixes explicitly, and compare a direct trend baseline with candidate
models. Report lead, missed red transitions and inspection burden **per unit**.
With only four observed training and one validation filter event, filter
forecast scores will remain exploratory; there is no observed event in the
held-out filter test prefixes on which to estimate red-transition recall.
