"""Spatio-temporal feature encoding shared by training and device inference."""

from __future__ import annotations

import calendar
import datetime as dt
import math
from typing import Sequence

import numpy as np


def parse_observation_date(value: str | dt.date | dt.datetime) -> dt.date:
    """Parse an iNaturalist-style timestamp or an ISO date."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    candidate = value.strip()
    if not candidate:
        raise ValueError("observation date cannot be empty")
    # Covers the legacy `%Y-%m-%d %H:%M:%S+00:00` data and ISO 8601 values.
    normalized = candidate.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(normalized).date()
    except ValueError:
        try:
            return dt.date.fromisoformat(candidate[:10])
        except ValueError as exc:
            raise ValueError(f"unsupported observation date: {value!r}") from exc


def year_fraction(value: str | dt.date | dt.datetime) -> float:
    """Return the 1-based day of year divided by the number of days in that year.

    This intentionally matches the encoding in the legacy geo-prior project.
    """
    date = parse_observation_date(value)
    days = 366 if calendar.isleap(date.year) else 365
    return date.timetuple().tm_yday / days


def encode_geo_features(
    latitude: float,
    longitude: float,
    observation_date: str | dt.date | dt.datetime,
) -> np.ndarray:
    """Encode longitude, latitude, and date as six sine/cosine features.

    Feature order is exactly `[sin(lon), cos(lon), sin(lat), cos(lat),
    sin(date), cos(date)]`, matching the original FCNet training code.
    """
    latitude = float(latitude)
    longitude = float(longitude)
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"latitude must be in [-90, 90], got {latitude}")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError(f"longitude must be in [-180, 180], got {longitude}")

    lat = latitude / 90.0
    lon = longitude / 180.0
    date = year_fraction(observation_date) * 2.0 - 1.0
    values: Sequence[float] = (
        math.sin(math.pi * lon),
        math.cos(math.pi * lon),
        math.sin(math.pi * lat),
        math.cos(math.pi * lat),
        math.sin(math.pi * date),
        math.cos(math.pi * date),
    )
    return np.asarray(values, dtype=np.float32)
