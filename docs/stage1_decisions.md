# Stage 1 decisions: prediction time, feature set, cost ratio, threshold

Decided by the stage 1 owner (Hoon) on 2026-11-25 for the stage 2 training pipeline. Open for team review
in the pull request that adds this file. Every number comes from `docs/decision_results.json`, produced by
`python -m src.eda.decision_analysis` on train and validation only. The test split is not loaded.

## Summary

| Decision | Choice |
|---|---|
| Prediction time | Score active bookings before arrival, using only fields fixed when the booking is made, as far as the data allows |
| Feature set | `DEFAULT_FEATURES` in `src/data/columns.py`: `STRICT` **without `deposit_type`** (18 columns) |
| Cost ratio | A false positive costs **2** empty rooms (r = 2); a false negative costs 1 |
| Decision threshold | The threshold that minimizes cost at r = 2 on validation, fixed before the test run. **0.77** for the exploratory model |
| Baselines the model must beat | Flagging nothing (333.7 per 1,000 bookings) and the untuned run below (255.5) |

## 1. Prediction time and what the data can support

The dataset paper says every row is a snapshot taken **the day before arrival**, for kept and canceled
bookings alike (Antonio, de Almeida, Nunes 2019, *Data in Brief*). A booking canceled weeks earlier is
therefore described by its state on that day, after the cancellation.

The model is meant to score bookings that are still active, before arrival. For that to be fair, a feature
must hold the same value at scoring time as in the snapshot. Fields set when the booking is made (lead time,
dates, stay length, party size, hotel, meal, segment, channel, customer type, guest history) meet this.
Fields that can change after booking, or after a cancellation, do not, and are removed where the evidence
shows a problem (section 2).

## 2. Feature set: `STRICT` without `deposit_type`

`DEFAULT_FEATURES` (18 columns): lead time, arrival week, day of month and month, weekend and week nights,
adults, children, babies, repeated guest, previous cancellations, ADR, special requests, hotel, meal,
market segment, distribution channel, customer type.

| Column | Why it is out |
|---|---|
| `reservation_status`, `reservation_status_date` | They are the outcome |
| `deposit_type` | Timing cannot be verified; see below |
| `country` | Nationality is often unknown until check-in (`docs/eda_findings.md`, section 5) |
| `required_car_parking_spaces` | Zero cancellations among 5,959 bookings with a parking request, the pattern of a field filled at check-in. Not confirmed with the hotel |
| `assigned_room_type`, `booking_changes` | Set or counted up to check-in or cancellation |
| `arrival_date_year` | Never repeats in the future, so it cannot generalize |

**Why `deposit_type` is out.** Non Refund bookings are canceled 99.2% of the time (11,768 rows in train and
validation). That is hard to square with what the label means, and the meaning cannot be checked:

1. The dataset paper computes the column from payments recorded before the arrival date, or before the
   cancellation date for canceled bookings: "Non Refund" when they cover the full cost of the stay. So a
   charge made at cancellation should not count. But the payment can come any time after the booking, and
   the data has no payment dates, so at scoring time the value may not be known yet. And fully prepaid,
   non-refundable bookings that are almost always canceled are an economic oddity the data does not explain.
2. A published explanation does not fit this data. The same authors report, for a different set of eight
   anonymous Portuguese hotels, that canceled non-refundable bookings were mostly fake online-travel-agency
   bookings from some Asian countries, made to support visa applications (Antonio, de Almeida, Nunes 2019,
   *Cornell Hospitality Quarterly*). Here, Non Refund bookings are 96.6% Portugal, 62.4% Groups and 35.0%
   Offline TA/TO, only 0.4% Online TA, and 93.3% are exact copies of another Non Refund row. Median lead
   time is 211 days against 47 for other bookings, and canceled ones are canceled a median 93 days before
   arrival against 27. This looks more like group or tour-operator blocks released before arrival than
   visa fraud, but the data cannot confirm either story.

Keeping a column whose meaning and timing are both unexplained would make every reported score depend on
it. We drop it from the default and report it as an ablation.

Cost of the choice: PR-AUC on validation falls from 0.7825 to 0.7328. The model still flags 905 of the
1,307 Non Refund validation bookings at the chosen threshold, from the other columns (segment, lead time),
so much of the signal survives without the column.

Special requests stay in, with a caveat: they can be added after booking, and canceled bookings may stop
receiving them. Removing them costs about 0.02 PR-AUC (`eda_findings.md`, section 7). Listed under
limitations.

## 3. Cost ratio r = 2

The action behind a positive prediction is assumed to be **targeted overbooking**: the room of a flagged
booking is sold again.

- **False negative** (a cancellation the model misses): the room stays empty. Cost 1, the unit.
- **False positive** (a flagged booking whose guest arrives): the hotel has no room and must "walk" the
  guest to another hotel, paying for the room, transport and goodwill. Cost r.

In eCornell course material on overbooking, Sheryl Kimes uses an example where walking a guest costs
3 times an empty room, and says some hotels use 2 times as a rule of thumb (eCornell #IMPACT blog, 2013).
We take **r = 2** and report r = 1 and r = 3 as sensitivity. r = 2 is an assumption, not a measured cost:
the data has no compensation or capacity figures.

## 4. Threshold rule

With perfectly calibrated probabilities, flagging pays off above r / (1 + r), which is 0.667 at r = 2. The
exploratory model is not calibrated well enough for that:

- Brier score 0.157; mean prediction 0.313 against an observed rate of 0.334.
- Middle deciles under-predict (decile 6: 0.255 predicted, 0.359 observed). Decile 9, where the threshold
  sits, over-predicts (0.653 predicted, 0.581 observed).
- Isotonic calibration does not fix it. Fitting the model on train before October 2016 and the calibration
  on October to December 2016 made validation cost at 0.667 worse (269.9 against 264.0 per 1,000 bookings)
  and lowered PR-AUC (0.700 against 0.729). The calibration months differ from the validation months.

So the threshold is **chosen on validation**: the value that minimizes cost at r = 2. For the exploratory
model that is **0.77**, and it is stable: 0.73 to 0.78 across bootstrap resamples (5th to 95th percentile).
The validation set exists for this kind of choice; the test split stays untouched until the production check.
The cost at the chosen threshold is slightly optimistic because it is measured on the same rows.

## 5. Evidence (exploratory model, validation)

Model: `HistGradientBoostingClassifier` with defaults on `DEFAULT_FEATURES`, fit on train, scored on the
12,828 validation bookings. Cost per 1,000 bookings, in empty rooms. Intervals: 300 bootstrap resamples.

![Validation cost by threshold](figures/fig7_cost_threshold.png)

| r | Lowest-cost threshold | Cost there | Cost at r/(1+r) | Flag nothing | Saving vs flag nothing at r/(1+r) [95% CI] |
|---|---|---|---|---|---|
| 1 | 0.52 | 231.4 | 234.3 (0.500) | 333.7 | 99.4 [92.4, 107.5] |
| **2** | **0.77** | **255.5** | 264.7 (0.667) | 333.7 | 69.1 [62.0, 76.7] |
| 3 | 0.78 | 262.3 | 266.2 (0.750) | 333.7 | 67.5 [60.8, 73.9] |

At r = 2 and threshold 0.77 the model flags 9.9% of bookings, with precision 0.929 and recall 0.276. That
cuts the cost of doing nothing by 23% (78.2 rooms per 1,000 bookings, interval 72.6 to 83.8).

Per validation month at 0.77: January precision 0.949, recall 0.298; February 0.948 and 0.244; March 0.902
and 0.287.

For reference only (both use the excluded column): the same model with `deposit_type` costs 229.3 at
r = 2 and 0.667, and the rule "flag every Non Refund booking" costs 233.5. These numbers show what the
column would add if it were safe, not targets to chase.

## 6. What this means for stage 2

1. Train on `DEFAULT_FEATURES`. Report `STRICT` (with `deposit_type`) only as an ablation, labeled as
   possibly leaking.
2. Log these metrics in MLflow for every run, on validation:
   - PR-AUC (ranking metric)
   - cost per 1,000 bookings at r = 2, at the run's own lowest-cost threshold, with the threshold, precision
     and recall
   - Brier score
3. The untuned model above (PR-AUC 0.7328, cost 255.5) is the baseline run. A champion should beat it on
   cost at r = 2, with the threshold fixed from validation before the test run.

## 7. Limitations

- **r is an industry rule of thumb**, not this hotel's cost. A walk only happens when the hotel is actually
  full, so the true false-positive cost is probably lower. At r = 1 the model saves about 100 rooms per
  1,000 bookings instead of about 70.
- **Cost is per booking**, not weighted by revenue. Long or expensive stays matter more in reality.
- **The day-before-arrival snapshot** affects every field that can change after booking. Special requests
  and ADR are kept but could differ between scoring time and the snapshot for canceled bookings.
- **Validation covers January to March 2017 only.** Cancellation rates vary by month (`eda_findings.md`,
  section 3), and the test months (April to August) are not in validation, so precision and recall at the
  chosen threshold will move. The production check on the test split will show by how much.

## Sources

- Antonio, N., de Almeida, A., Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41-49.
  https://doi.org/10.1016/j.dib.2018.11.126 (author copy:
  https://repositorio.iscte-iul.pt/bitstream/10071/16929/1/Hotelbookingdemanddatasets.pdf)
- Antonio, N., de Almeida, A., Nunes, L. (2019). Big Data in Hotel Revenue Management: Exploring
  Cancellation Drivers to Gain Insights Into Booking Cancellation Behavior. *Cornell Hospitality
  Quarterly*. Author postprint:
  https://repositorio.iscte-iul.pt/bitstream/10071/18184/1/BigDataCancellationPrediction_posprint.pdf
- Kimes, S. (2013). The cheapest and best approach to overbooking. eCornell #IMPACT blog.
  https://ecornell-impact.cornell.edu/the-cheapest-and-best-approach-to-overbooking/
