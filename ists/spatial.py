from typing import Dict, List, Tuple, Literal
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from .preprocessing import time_encoding


def array_mapping(id_array: np.ndarray, time_array: np.ndarray, dist_array: np.ndarray) -> Dict[str, pd.DataFrame]:
    maps = pd.DataFrame({
        'Id': id_array,
        'Time': time_array,
        'Null': np.max(dist_array, axis=1),
        'Idx': np.arange(len(id_array))
    })
    maps = dict(list(maps.groupby('Id')))
    for k, df in maps.items():
        k_map = pd.DataFrame({'Idx': df['Idx'].values, 'Null': df['Null'].values}, index=df['Time'].values)
        maps[k] = k_map
    return maps


def prepare_spatial_data(
        x_array: np.ndarray,
        id_array: np.ndarray,
        time_array: np.ndarray,
        dist_x_array: np.ndarray,
        num_past: int,
        num_spt: int,
        spt_dict: Dict[str, pd.Series],
        max_dist_th: float = None,
        max_null_th: int = None,
) -> Tuple[List[np.array], np.ndarray]:
    max_null_th = max_null_th if max_null_th else np.float('inf')
    max_dist_th = max_dist_th if max_dist_th else np.float('inf')

    spt_array = [[] for _ in range(num_spt)]
    mask = []
    arr_mapping = array_mapping(id_array, time_array, dist_x_array)
    # Keep only spatial id that can be used
    spt_dict = {k: s.index[(s.index.isin(id_array)) & (s <= max_dist_th)] for k, s in spt_dict.items()}
    # for rid, rtime in tqdm(zip(id_array, time_array)):
    for rid, rtime in zip(id_array, time_array):
        # Keep only spatial id that can be used
        spt_ids = spt_dict[rid]
        spt_count = 0
        spt_mem = []
        for spt_id in spt_ids:
            if rtime in arr_mapping[spt_id].index and arr_mapping[spt_id].loc[rtime, 'Null'] <= max_null_th :
                spt_count += 1
                spt_mem.append(arr_mapping[spt_id].loc[rtime, 'Idx'])

            if spt_count == num_spt:
                break

        assert spt_count <= num_spt
        if spt_count == num_spt:
            mask.append(True)
            for i in range(num_spt):
                spt_array[i].append(x_array[spt_mem[i]])
        else:
            mask.append(False)

    spt_res_array = []
    for i in range(num_spt):
        spt_res_array.append(np.array(spt_array[i])[:, -num_past:, :])
    return spt_res_array, np.array(mask)


def find_last_date_idx(df: pd.DataFrame, date: datetime.date, th: int = 7) -> int:
    df_slice = df[df.index <= date]
    if df_slice.empty:
        return -1

    last_date = df_slice.index[-1]
    last_date_idx = df.index.get_loc(last_date)

    if (date - last_date ).days > th:
        return -1
    else:
        return last_date_idx


def prepare_exogenous_data(
        id_array: np.ndarray,
        time_array: np.ndarray,
        exg_dict: Dict[str, pd.DataFrame],
        num_past: int,
        features: List[str],
        time_feats: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    exg_array = []
    mask = []

    for k, df in exg_dict.items():
        df = time_encoding(df, codes=time_feats)
        df = df[features + time_feats]
        exg_dict[k] = df

    # for rid, rtime in tqdm(zip(id_array, time_array)):
    for rid, rtime in zip(id_array, time_array):
        df: pd.DataFrame = exg_dict[rid]
        # df = time_encoding(df, codes=time_feats)
        # df = df[features + time_feats]

        right = find_last_date_idx(df, date=rtime)
        if right != -1:  # [features]
            right = right + 1
            left = right - num_past
            if left >= 0:
                mask.append(True)
                exg_array.append(df.iloc[left:right].values)
            else:
                mask.append(False)
        else:
            mask.append(False)

    exg_array = np.array(exg_array, dtype=float)
    return exg_array, np.array(mask)
