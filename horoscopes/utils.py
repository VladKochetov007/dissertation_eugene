"""
Shared utilities: zodiac assignment, trait scoring, data loading.
"""
import numpy as np
import pandas as pd
from typing import Optional


# --- Zodiac boundaries (day of year, 1-indexed, non-leap) ---
# Each sign: (name, start_doy, end_doy)
ZODIAC_SIGNS = [
    ("Capricorn",    1,  19),   # Jan 1 – Jan 19  (continues from Dec 22)
    ("Aquarius",    20,  49),   # Jan 20 – Feb 18
    ("Pisces",      50,  79),   # Feb 19 – Mar 20
    ("Aries",       80, 109),   # Mar 21 – Apr 19
    ("Taurus",     110, 140),   # Apr 20 – May 20
    ("Gemini",     141, 171),   # May 21 – Jun 20
    ("Cancer",     172, 203),   # Jun 21 – Jul 22
    ("Leo",        204, 234),   # Jul 23 – Aug 22
    ("Virgo",      235, 265),   # Aug 23 – Sep 22
    ("Libra",      266, 295),   # Sep 23 – Oct 22
    ("Scorpio",    296, 325),   # Oct 23 – Nov 21
    ("Sagittarius",326, 355),   # Nov 22 – Dec 21
    # Capricorn tail: 356-365 → maps to Capricorn (index 0)
]
ZODIAC_NAMES = [z[0] for z in ZODIAC_SIGNS]
N_SIGNS = 12
CHANCE_BASELINE = 1.0 / N_SIGNS  # 0.0833...

# Boundary cut-points (11 internal boundaries, in day-of-year space)
# These are the start-days of signs 1-11 (Aquarius through Sagittarius)
ZODIAC_BOUNDARIES = [z[1] for z in ZODIAC_SIGNS[1:]]  # [20, 50, 80, 110, 141, 172, 204, 235, 266, 296, 326]

BIG5_TRAITS = ["E", "A", "C", "N", "O"]  # Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness


def doy_to_zodiac(doy: np.ndarray) -> np.ndarray:
    """
    Convert day-of-year (1-365) to zodiac sign index (0-11).
    Capricorn spans 356-365 and 1-19 (index 0).
    """
    doy = np.asarray(doy)
    signs = np.where(doy >= 356, 0, -1)  # Capricorn tail
    for i, (_, start, end) in enumerate(ZODIAC_SIGNS):
        mask = (doy >= start) & (doy <= end) & (signs == -1)
        signs = np.where(mask, i, signs)
    return signs


def doy_to_zodiac_name(doy: np.ndarray) -> np.ndarray:
    idx = doy_to_zodiac(doy)
    return np.array(ZODIAC_NAMES)[idx]


def partition_doy(doy: np.ndarray, boundaries: list[int]) -> np.ndarray:
    """
    Assign day-of-year values to segments given a sorted list of
    boundary start-days (length k-1 gives k segments).
    Returns segment index (0-indexed).
    """
    doy = np.asarray(doy)
    seg = np.zeros(len(doy), dtype=int)
    for i, b in enumerate(sorted(boundaries)):
        seg[doy >= b] = i + 1
    return seg


def score_big5(df: pd.DataFrame, instrument: str = "TIPI") -> pd.DataFrame:
    """
    Compute Big Five domain scores from raw item responses.
    Supports TIPI (10 items) and IPIP-FFM (50 items, EXT/EST/AGR/CSN/OPN).
    Returns df with columns E, A, C, N, O added.
    """
    if instrument == "TIPI":
        # TIPI scoring: items 1-10, reverse-score even items
        # E = (TIPI1 + (8-TIPI6)) / 2
        # A = (TIPI7 + (8-TIPI2)) / 2
        # C = (TIPI3 + (8-TIPI8)) / 2
        # N = (TIPI4 + (8-TIPI9)) / 2
        # O = (TIPI5 + (8-TIPI10)) / 2
        df = df.copy()
        df["E"] = (df["TIPI1"] + (8 - df["TIPI6"])) / 2
        df["A"] = (df["TIPI7"] + (8 - df["TIPI2"])) / 2
        df["C"] = (df["TIPI3"] + (8 - df["TIPI8"])) / 2
        df["N"] = (df["TIPI4"] + (8 - df["TIPI9"])) / 2
        df["O"] = (df["TIPI5"] + (8 - df["TIPI10"])) / 2
    elif instrument == "IPIP-FFM":
        # 10 items per trait, some reverse-scored (indicated by Q suffix or sign convention)
        # Using openpsychometrics IPIP-FFM variable naming
        df = df.copy()
        for trait, cols_pos, cols_neg in [
            ("E",
             ["EXT1", "EXT3", "EXT5", "EXT7", "EXT9"],
             ["EXT2", "EXT4", "EXT6", "EXT8", "EXT10"]),
            ("A",
             ["AGR2", "AGR4", "AGR6", "AGR8", "AGR9"],
             ["AGR1", "AGR3", "AGR5", "AGR7", "AGR10"]),
            ("C",
             ["CSN1", "CSN3", "CSN5", "CSN7", "CSN9"],
             ["CSN2", "CSN4", "CSN6", "CSN8", "CSN10"]),
            ("N",
             ["EST2", "EST4"],
             ["EST1", "EST3", "EST5", "EST6", "EST7", "EST8", "EST9", "EST10"]),
            ("O",
             ["OPN1", "OPN3", "OPN5", "OPN7", "OPN8", "OPN9", "OPN10"],
             ["OPN2", "OPN4", "OPN6"]),
        ]:
            pos = df[cols_pos].mean(axis=1)
            neg = (6 - df[cols_neg]).mean(axis=1)
            df[trait] = (pos * len(cols_pos) + neg * len(cols_neg)) / 10
    return df


def load_real_data(path: str, instrument: str = "TIPI") -> Optional[pd.DataFrame]:
    """
    Load a real dataset CSV. Expects columns: doy (or birth_month + birth_day),
    plus raw personality item columns. Returns df with E,A,C,N,O and zodiac columns.
    """
    df = pd.read_csv(path, low_memory=False)

    # Derive day-of-year if not present
    if "doy" not in df.columns:
        if "birth_month" in df.columns and "birth_day" in df.columns:
            # Approximate: ignore leap years
            month_offsets = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
            df["doy"] = df["birth_month"].apply(
                lambda m: month_offsets[int(m) - 1]
            ) + df["birth_day"]
        else:
            raise ValueError("Dataset must have 'doy' or ('birth_month' + 'birth_day') columns")

    df = score_big5(df, instrument)
    df["zodiac_idx"] = doy_to_zodiac(df["doy"].values)
    df["zodiac"] = doy_to_zodiac_name(df["doy"].values)
    return df
