from datetime import datetime


W1_START = datetime(2025, 1, 1)
W1_END = datetime(2025, 6, 30, 23, 59, 59)

W2_START = datetime(2025, 7, 1)
W2_END = datetime(2025, 12, 31, 23, 59, 59)

W3_START = datetime(2026, 1, 1)
W3_END = datetime(2026, 6, 30, 23, 59, 59)


def get_time_window(created_utc: float) -> str:
    dt = datetime.utcfromtimestamp(created_utc)

    if W1_START <= dt <= W1_END:
        return "W1"

    if W2_START <= dt <= W2_END:
        return "W2"

    if W3_START <= dt <= W3_END:
        return "W3"

    return "OUT_OF_RANGE"