from __future__ import annotations

from typing import Tuple
import pandas as pd


def time_split_last_n(
    data: pd.DataFrame,
    last_n: int = 1,
    user_col: str = "user_id",
    time_col: str = "datetime",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-last-n-out split per user based on timestamp."""
    if time_col not in data.columns:
        raise ValueError(f"Column '{time_col}' is required for time split")

    df = data.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values([user_col, time_col])

    test = df.groupby(user_col, group_keys=False).tail(last_n).copy()
    train = df.drop(index=test.index).copy()

    # Keep only users that still have at least one train interaction.
    good_users = train[user_col].unique()
    test = test[test[user_col].isin(good_users)].copy()

    return train.reset_index(drop=True), test.reset_index(drop=True)
