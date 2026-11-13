"""Column groups for the hotel bookings data (TidyTuesday 2020-02-11 version).

The groups encode what the EDA found (see docs/eda_findings.md). Timing statements
come from the data article (Antonio et al. 2019): attribute values describe the
booking as of the day before arrival, BookingChanges counts amendments until
check-in or cancellation, and hotels often do not know a guest's nationality
until check-in. Which columns are known when a booking is made cannot be proven
from the data alone, so the groups below are decisions, not facts.
"""

TARGET = "is_canceled"

NUMERIC = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]
CATEGORICAL = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "deposit_type",
    "customer_type",
]

# Arrival-date parts. Their distribution differs between any two time windows by design.
CALENDAR = [
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
]

# Recorded after the outcome is known: perfectly separate the target. Never a feature.
OUTCOME_COLUMNS = ["reservation_status", "reservation_status_date"]

# Plausibly only known at or after check-in (country: article says nationality is often
# unknown until check-in; parking: the data show 0 cancellations among requests).
TIMING_RISK = ["country", "required_car_parking_spaces"]

# Defined at or after check-in/cancellation (article): amendments counted until
# check-in or cancellation; assigned room can differ from reserved for hotel reasons.
POST_BOOKING = ["assigned_room_type", "booking_changes"]

# The feature list used in the earlier project proposal.
PROPOSAL = NUMERIC + CATEGORICAL

# Proposed default: proposal list without the timing-risk columns and without the year.
STRICT = [c for c in PROPOSAL if c not in TIMING_RISK and c != "arrival_date_year"]

FEATURE_SETS = {
    "proposal": PROPOSAL,
    "strict": STRICT,
    "strict + assigned room, booking changes": STRICT
    + ["assigned_room_type", "booking_changes", "reserved_room_type"],
    "strict + country": STRICT + ["country"],
    "strict + parking": STRICT + ["required_car_parking_spaces"],
}
