from typing import Dict, List, Tuple
from datetime import datetime, timedelta

import pandas as pd

from ..piezo.read import ContextType, create_ts_dict, create_spatial_matrix
from ...preparation import reindex_ts
from ..utils import insert_null_values


def read_ushcn(filename: str, id_col: str, date_col: str, cols: List[str]) -> Dict[str, pd.DataFrame]:
    # Read ushcn dataset
    df = pd.read_csv(filename)

    # Transform timestamp column into datetime object
    df[date_col] = df[date_col].apply(lambda x: timedelta(days=x) + datetime(year=1950, month=1, day=1))
    df[date_col] = pd.to_datetime(df[date_col]).dt.date

    # Split time-series based on date_col and keep only the selected cols
    ts_dict = create_ts_dict(df=df, id_col=id_col, date_col=date_col, cols=cols)

    # Reindex time-series with daily frequency
    ts_dict = {k: reindex_ts(df, 'D') for k, df in ts_dict.items()}

    return ts_dict


def extract_ushcn_context(filename: str, id_col: str, x_col: str, y_col: str) -> Dict[str, ContextType]:
    # Read ushcn dataset
    df = pd.read_csv(filename)

    # Create an empty dict to save x and y coordinates for each id
    ctx_dict: Dict[str, ContextType] = {}
    # Create a dictionary where for each id is associated its time-series
    ts_dict: Dict[str, pd.DataFrame] = dict(list(df.groupby(id_col)))
    for k, df_k in ts_dict.items():
        # Check coordinates uniqueness
        assert df_k[x_col].duplicated(keep=False).all(), f'Different x coordinates for series {k}'
        assert df_k[y_col].duplicated(keep=False).all(), f'Different y coordinates for series {k}'

        # Extract coordinates from k series
        ctx_dict[k] = {
            'x': df_k[x_col].iloc[0],
            'y': df_k[y_col].iloc[0],
        }
    return ctx_dict


def load_ushcn_data(
        ts_filename: str,
        subset_filename: str = None,
        nan_percentage: float = 0
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
    # Read irregular ushcn time series
    ts_dict = read_ushcn(
        filename=ts_filename,
        id_col='UNIQUE_ID',
        date_col='TIME_STAMP',
        cols=["SNOW", "SNWD", "PRCP", "TMAX", "TMIN"],
    )

    # Filter based on a subset if any
    if subset_filename:
        subset = pd.read_csv(subset_filename)['UNIQUE_ID'].to_list()
        ts_dict = {k: ts_dict[k] for k in subset if k in ts_dict}

    # Extract coordinates from ushcn series
    ctx_dict = extract_ushcn_context(filename=ts_filename, id_col='UNIQUE_ID', x_col='X', y_col='Y')

    # Remove context information without time-series
    keys = list(ctx_dict.keys())
    for k in keys:
        if k not in ts_dict:
            # print(k)
            ctx_dict.pop(k)

    # Create a copy of exogenous series from the raw time-series dict
    exg_dict = {
        k: df[["SNOW", "SNWD", "PRCP", "TMAX", "TMIN"]].ffill().dropna(axis=0, how='any')
        for k, df in ts_dict.items()
    }

    # Create distance matrix for each pair of irregular time series by computing the haversine distance
    dist_matrix = create_spatial_matrix(ctx_dict, with_haversine=True)
    spt_dict = {}
    for k in ts_dict.keys():
        dists = dist_matrix.loc[k]
        dists = dists.drop(k)
        dists = dists.sort_values(ascending=True)
        spt_dict[k] = dists

    # Loop through the time-series and insert NaN values at the random indices
    if nan_percentage > 0:
        ts_dict = {
            k: insert_null_values(ts, nan_percentage, cols=["TMAX"])
            for k, ts in ts_dict.items()
        }

    return ts_dict, exg_dict, spt_dict
