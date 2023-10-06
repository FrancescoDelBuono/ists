from typing import Dict, List, Tuple, Literal, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .preprocessing import time_encoding


def reindex_ts(ts: pd.DataFrame, freq: Literal['M', 'W', 'D']):
    min_dt, max_dt = ts.index.min(), ts.index.max()  # Read min max date
    # dt_name = ts.index.name  # Date Index name
    # Create a new monthly or week index
    new_index = pd.date_range(start=min_dt, end=max_dt, freq=freq).date
    # Check index constraint
    assert np.isin(ts.index, new_index).all()
    # Reindex the time-series DataFrame
    ts = ts.reindex(new_index)
    return ts


def drop_first_nan_rows(df: pd.DataFrame, cols: list, reverse: bool) -> pd.DataFrame:
    def count_first_nan(s: pd.Series):
        """ Count the number of trues/nan at the start of the series"""
        is_nan_arr = s.values  # s.isnull().values  # Get array of booleans: True is nan; and, False is not nan
        if reverse:
            is_nan_arr = is_nan_arr[::-1]

        # Consider only the first max_start_drop rows
        size = len(is_nan_arr)
        num_nan = size  # Set not nan idx to max size
        for i, is_nan in enumerate(is_nan_arr[:size]):
            # Check if is not nan
            if not is_nan:
                # Save the position of the first not nan value
                num_nan = i
                break
        return num_nan

    # Compute the number of consecutive starting rows with at least on nan
    idx = count_first_nan(pd.isnull(df[cols]).any(axis=1))
    if idx > 0:
        if not reverse:
            # Remove first nan rows
            df = df.drop(df.index[:idx])
        elif reverse and idx > 0:
            # Remove last nan rows
            df = df.drop(df.index[-idx:])

    return df


def null_distance_array(is_null: np.ndarray, method: Literal['log', 'lin'] = 'lin', max_dist: int = None) -> np.ndarray:
    # Initialize arrays
    distance_array = np.zeros(len(is_null))
    last_observed_index = -1
    for i, val in enumerate(is_null):
        if not val:
            # Not null value
            # Set distance_array to 0
            distance_array[i] = 0
            # Reset last_observed_index
            last_observed_index = i
        else:
            # Compute distance from last_observed_index
            if last_observed_index >= 0:
                if method == 'lin':
                    # Linear distance
                    distance_array[i] = i - last_observed_index
                    if max_dist:
                        distance_array[i] /= max_dist
                elif method == 'log':
                    # Log linear distance
                    distance_array[i] = np.log(1 + i - last_observed_index)
                    if max_dist:
                        distance_array[i] /= np.log(max_dist)
                else:
                    raise ValueError("Invalid method option. Must be 'log', 'lin'.")
            else:
                raise ValueError('Input array start with null values')

    return distance_array


def null_indicator(ts: pd.DataFrame, method: Literal['code_bool', 'code_lin', 'bool', 'log', 'lin'] = 'bool',
                   max_dist: int = None) -> np.ndarray:
    cond_null = pd.isnull(ts).any(axis=1).values

    if method == 'bool':
        null_array = cond_null.astype(int)
    elif method == 'lin':
        null_array = null_distance_array(cond_null, method=method, max_dist=max_dist)
    elif method == 'log':
        null_array = null_distance_array(cond_null, method=method, max_dist=max_dist)
    elif method == 'code_lin':
        null_array = null_distance_array(cond_null, method='lin', max_dist=None)
    elif method == 'code_bool':
        null_array = cond_null.astype(int)
    else:
        raise ValueError("Invalid method option. Must be 'code_bool', 'code_lin', 'log', 'lin', or 'bool.")
    if (method == 'bool' or max_dist) and not method.startswith('code_'):
        null_array = 1 - null_array

    return null_array


def find_nearest_label(ts_label: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    label_isnull = ts_label.isnull().values
    distance_fw = null_distance_array(label_isnull, method='lin', max_dist=None)
    distance_bw = null_distance_array(label_isnull[::-1], method='lin', max_dist=None)[::-1]
    label_fw = ts_label.ffill().values
    label_bw = ts_label.bfill().values
    min_distance = np.where(distance_fw < distance_bw, distance_fw, distance_bw)
    nearest_label = np.where(distance_fw < distance_bw, label_fw, label_bw)

    return nearest_label, min_distance


def sliding_window(
        ts: pd.DataFrame,
        label_col: str,
        features: List[str],
        num_past: int,
        num_fut: int,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    # # Make the label col the first in dataframe
    # cols = ts.columns[ts.columns != label_col].to_list()
    # ts = ts[[label_col] + cols]

    # Fix stride to 1
    stride = 1

    # Save date col name and transform it into column
    date_col = ts.index.name
    ts = ts.reset_index()

    # Define null distance index
    ts['__NULL__'] = null_distance_array(ts[label_col].isnull().values, method='lin', max_dist=None)

    # Find for each null label the nearest not null value and the distance from it in forward and backward
    new_label, dist_label = find_nearest_label(ts[label_col])
    ts['__LABEL__'] = new_label
    ts['__DISTANCE__'] = dist_label

    # Define window size
    window = num_past + num_fut
    rows, cols = ts.shape

    # Check minimum rows constraints
    if rows <= window:
        return None

    # New number of rows after the sliding window
    new_rows = 1 + (rows - window) // stride
    # Empty matrix to save sliding window segments
    matrix = np.zeros((new_rows, window, cols), dtype=object)

    # Sliding window process
    data = ts.values
    for i in range(new_rows):
        left = i * stride
        right = left + window
        matrix[i, :, :] = data[left:right, :]

    cond_x = np.isin(ts.columns, features)
    cond_y = ts.columns == '__LABEL__'
    cond_date = ts.columns == date_col
    cond_dist_x = ts.columns == '__NULL__'
    cond_dist_y = ts.columns == '__DISTANCE__'

    x = matrix[:, :num_past, cond_x]
    y = matrix[:, window - 1, cond_y]
    time_mask = matrix[:, [0, num_past - 1, -1], cond_date]
    dist_x = matrix[:, :num_past, cond_dist_x]
    dist_y = matrix[:, window - 1, cond_dist_y]

    # y = np.squeeze(y, axis=2)
    # time_y = np.squeeze(time_y, axis=2)

    return x.astype(float), y.astype(float), time_mask, dist_x.astype(float), dist_y.astype(float)


def define_feature_mask(base_features: List[str], null_feat: str = None, time_feats: List[str] = None) -> List[int]:
    # Return the type of feature (0: raw, 1: null encoding, 2: time encoding) in each timestamp
    features_mask = [0 for _ in base_features]
    if null_feat and null_feat in ['code_lin', 'code_bool']:
        features_mask += [1]
    elif null_feat and null_feat not in ['code_lin', 'code_bool']:
        features_mask += [0]
    if time_feats:
        features_mask += [2 for _ in time_feats]
    return features_mask


def get_null_max_size(x: np.ndarray, feature_mask: List[int]) -> Optional[int]:
    if 1 not in feature_mask:
        # No null feature inside input data
        return None
    cond = np.array(feature_mask) == 1
    max_null = int(np.max(x[:, :, cond]))
    return max_null + 1


def get_list_null_max_size(arr_list: List[np.ndarray], feature_mask: List[int]) -> Optional[int]:
    if 1 not in feature_mask:
        # No null feature inside input data
        return None
    null_max_sizes = [get_null_max_size(arr, feature_mask) for arr in arr_list]
    return np.max(null_max_sizes)


def prepare_data(
        ts_dict: Dict[str, pd.DataFrame],
        num_past: int,
        num_fut: int,
        features: List[str],
        label_col: str,
        freq: Literal['M', 'W', 'D'] = 'D',
        null_feat: Literal['code_bool', 'code_lin', 'bool', 'log', 'lin'] = None,
        null_max_dist: int = None,
        time_feats: List[str] = None,
        with_fill: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Empty array for saving x, y, time mask, and id for each record
    x_array = []
    y_array = []
    time_array = []
    dist_x_array = []
    dist_y_array = []
    id_array = []

    assert label_col in features, f'Error, label {label_col} must be in the selected features'
    features.remove(label_col)
    features.insert(0, label_col)

    # Iterate each time-series
    # for k, ts in tqdm(ts_dict.items()):
    for k, ts in ts_dict.items():
        # Reorder time-series column in order to have the label at the start
        ts = ts[features].copy()

        new_features = []
        # Reindex time index
        ts_new = reindex_ts(ts, freq=freq)

        # Drop nan label rows at the start
        ts_new = drop_first_nan_rows(ts_new, features, reverse=False)
        # Drop nan rows at the end
        ts_new = drop_first_nan_rows(ts_new, [label_col], reverse=True)

        # Null indicator feature
        if null_feat:
            ts_new['NullFeat'] = null_indicator(ts_new, method=null_feat, max_dist=null_max_dist)
            new_features += ['NullFeat']
        if null_max_dist:
            ts_new['__NULL__'] = null_indicator(ts_new, method='code_lin')
            # new_features += ['__NULL__']

        # Time Encoding
        if time_feats:
            ts_new = time_encoding(ts_new, codes=time_feats)
            new_features += time_feats

        # Forward fill null values
        ts_new['__LABEL__'] = ts_new[label_col]
        if with_fill:
            ts_new[features + new_features] = ts_new[features + new_features].ffill()
        # Sliding window step
        blob = sliding_window(ts_new, '__LABEL__', features + new_features, num_past, num_fut)
        if blob is None:
            continue

        x_array += [blob[0]]
        y_array += [blob[1]]
        time_array += [blob[2]]
        dist_x_array += [blob[3]]
        dist_y_array += [blob[4]]
        id_array += [k] * len(blob[0])

    x_array = np.concatenate(x_array)
    y_array = np.concatenate(y_array)
    time_array = np.concatenate(time_array)
    dist_x_array = np.concatenate(dist_x_array)
    dist_y_array = np.concatenate(dist_y_array)
    id_array = np.array(id_array)

    # y_array = np.squeeze(y_array)
    dist_x_array = np.squeeze(dist_x_array)
    dist_y_array = np.squeeze(dist_y_array)

    return x_array, y_array, time_array, dist_x_array, dist_y_array, id_array


def filter_data(
        x_array: np.ndarray,
        y_array: np.ndarray,
        time_array: np.ndarray,
        dist_x_array: np.ndarray,
        dist_y_array: np.ndarray,
        id_array: np.ndarray,
        spt_array: List[np.ndarray],
        train_start: str,
        max_null_th: int = None,
        max_label_th: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    # cond 1 predicted value null distance not greater than a threshold -> dist_y_array <= max_label_th
    cond1 = dist_y_array <= max_label_th
    # cond 2 input array max null distance not greater than a threshold -> np.min(dist_x_array) <= max_null_th
    cond2 = np.max(dist_x_array, axis=1) <= max_null_th
    # cond 3 training start date constraint
    cond3 = (time_array[:, 0] > pd.to_datetime(train_start).date())
    # cond = cond 1 & cond 2 & cond 3
    mask = cond1 & cond2 & cond3
    x_array = x_array[mask]
    y_array = y_array[mask]
    time_array = time_array[mask]
    dist_x_array = dist_x_array[mask]
    dist_y_array = dist_y_array[mask]
    id_array = id_array[mask]
    spt_array = [arr[mask] for arr in spt_array]

    return x_array, y_array, time_array, dist_x_array, dist_y_array, id_array, spt_array


def prepare_train_test(
        x_array: np.ndarray,
        y_array: np.ndarray,
        time_array: np.ndarray,
        dist_x_array: np.ndarray,
        dist_y_array: np.ndarray,
        id_array: np.ndarray,
        spt_array: List[np.ndarray],
        exg_array: np.ndarray,
        test_start: str,
) -> dict:
    is_train = time_array[:, -1] < pd.to_datetime(test_start).date()
    is_test = (dist_y_array == 0) & (~is_train)
    res = {
        'x_train': x_array[is_train],
        'y_train': y_array[is_train],
        'time_train': time_array[is_train],
        'dist_x_train': dist_x_array[is_train],
        'dist_y_train': dist_y_array[is_train],
        'id_train': id_array[is_train],
        'spt_train': [arr[is_train] for arr in spt_array],
        'exg_train': exg_array[is_train],
        'x_test': x_array[is_test],
        'y_test': y_array[is_test],
        'time_test': time_array[is_test],
        'dist_x_test': dist_x_array[is_test],
        'dist_y_test': dist_y_array[is_test],
        'id_test': id_array[is_test],
        'spt_test': [arr[is_test] for arr in spt_array],
        'exg_test': exg_array[is_test],
    }
    return res
