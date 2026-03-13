from __future__ import annotations

from typing import Optional
import pandas as pd


def build_edges(
    data: pd.DataFrame,
    threshold: Optional[float] = 4.0,
    user_col: str = "user_id",
    item_col: str = "movie_id",
    rating_col: str = "rating",
    time_col: str = "datetime",
) -> pd.DataFrame:
    """Convert ratings into positive user-item interactions."""
    df = data.copy()

    if rating_col in df.columns and threshold is not None:
        df = df[df[rating_col] >= float(threshold)]

    cols = [user_col, item_col]
    if time_col in df.columns:
        cols.append(time_col)

    pos = df[cols].dropna().copy()
    pos[user_col] = pos[user_col].astype(int)
    pos[item_col] = pos[item_col].astype(int)

    if time_col in pos.columns:
        pos[time_col] = pd.to_datetime(pos[time_col], utc=True, errors="coerce")
        pos = pos.dropna(subset=[time_col])
        pos = pos.drop_duplicates(subset=[user_col, item_col, time_col])
    else:
        pos = pos.drop_duplicates(subset=[user_col, item_col])

    return pos.reset_index(drop=True)



def filter_users_min_pos(
    data: pd.DataFrame,
    min_pos: int = 5,
    user_col: str = "user_id",
) -> pd.DataFrame:
    """Keep only users with at least min_pos positive interactions."""
    user_counts = data.groupby(user_col).size()
    good_users = user_counts[user_counts >= min_pos].index
    return data[data[user_col].isin(good_users)].copy().reset_index(drop=True)



def interactions_stats(
    df_pos: pd.DataFrame,
    user_col: str = "user_id",
) -> pd.Series:
    cnt = df_pos.groupby(user_col).size()
    return pd.Series({
        "users_total": int(cnt.shape[0]),
        "min": int(cnt.min()) if len(cnt) else 0,
        "p25": float(cnt.quantile(0.25)) if len(cnt) else 0.0,
        "median": float(cnt.median()) if len(cnt) else 0.0,
        "p75": float(cnt.quantile(0.75)) if len(cnt) else 0.0,
        "max": int(cnt.max()) if len(cnt) else 0,
        "users_lt5": int((cnt < 5).sum()),
        "users_lt10": int((cnt < 10).sum()),
        "users_lt20": int((cnt < 20).sum()),
    })
