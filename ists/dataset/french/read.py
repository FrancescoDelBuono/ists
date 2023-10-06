from typing import Dict, List, Tuple

import pandas as pd

from ..piezo.read import create_ts_dict, read_context, create_spatial_matrix
from ..utils import insert_null_values


def read_french(filename: str, id_col: str, date_col: str, cols: List[str]) -> Dict[str, pd.DataFrame]:
    # Read ushcn dataset
    df = pd.read_csv(filename)

    # Transform timestamp column into datetime object
    df[date_col] = pd.to_datetime(df[date_col]).dt.date

    # Split time-series based on date_col and keep only the selected cols
    ts_dict = create_ts_dict(df=df, id_col=id_col, date_col=date_col, cols=cols)

    return ts_dict


def load_frenchpiezo_data(
        ts_filename: str,
        context_filename: str,
        subset_filename: str = None,
        nan_percentage: float = 0
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
    # Read irregular french piezo time series
    ts_dict = read_french(ts_filename, id_col='bss', date_col='time', cols=['tp', 'e', 'p'])

    # Filter based on a subset if any
    if subset_filename:
        subset = pd.read_csv(subset_filename)['bss'].to_list()
        ts_dict = {k: ts_dict[k] for k in subset if k in ts_dict}

    # Read time series context (i.e. coordinates)
    ctx_dict = read_context(context_filename, id_col='bss', x_col='x', y_col='y')

    # Remove context information without time-series
    keys = list(ctx_dict.keys())
    for k in keys:
        if k not in ts_dict:
            # print(k)
            ctx_dict.pop(k)

    # Create a copy of exogenous series from the raw time-series dict
    exg_dict = {
        k: df[["tp", "e"]].ffill().dropna(axis=0, how='any')
        for k, df in ts_dict.items()
    }

    # Create distance matrix for each pair of irregular time series
    dist_matrix = create_spatial_matrix(ctx_dict, with_haversine=False)
    spt_dict = {}
    for k in ts_dict.keys():
        dists = dist_matrix.loc[k]
        dists = dists.drop(k)
        dists = dists.sort_values(ascending=True)
        spt_dict[k] = dists

    # Loop through the time-series and insert NaN values at the random indices
    if nan_percentage > 0:
        ts_dict = {
            k: insert_null_values(ts, nan_percentage, cols=['p'])
            for k, ts in ts_dict.items()
        }

    return ts_dict, exg_dict, spt_dict
