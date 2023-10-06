from typing import Dict, Tuple, Callable
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


def tp_sum(df: pd.DataFrame) -> pd.Series:
    res = pd.Series({'tp': df['tp'].sum()})
    res['__DATE__'] = df['__DATE__'].iloc[-1]
    return res


def t2m_stats(df: pd.DataFrame) -> pd.Series:
    res = pd.Series({'t2m_min': df['t2m'].min(), 't2m_max': df['t2m'].max(), 't2m_avg': df['t2m'].mean()})
    res['__DATE__'] = df['__DATE__'].iloc[-1]
    return res


def compress_ts(df: pd.DataFrame, method: Callable[[pd.DataFrame], pd.Series], freq: str = 'D') -> pd.DataFrame:
    # Create date columns without time
    df['__DATE__'] = pd.to_datetime(df.index).date
    if freq == 'D':
        df['__YEAR__'] = df.index.map(lambda x: x.year)
        df['__GROUP__'] = df.index.map(lambda x: x.date)
    elif freq == 'M':
        df['__YEAR__'] = df.index.map(lambda x: x.year)
        df['__GROUP__'] = df.index.map(lambda x: x.month)
    elif freq == 'W':
        df['__YEAR__'] = df.index.map(lambda x: x.isocalendar()[0])
        df['__GROUP__'] = df.index.map(lambda x: x.isocalendar()[1])
    else:
        raise ValueError('freq "{}" is not supported'.format(freq))

    df = df.groupby(['__YEAR__', '__GROUP__']).apply(lambda arr: method(arr))
    df.index = df['__DATE__'].values
    df = df.sort_index(ascending=True)
    df = df.drop('__DATE__', axis=1)
    return df


def combine_tp_files(tp1_filename: str, tp2_filename: str) -> Dict[Tuple[float, float], pd.DataFrame]:
    with open(tp1_filename, "rb") as f:
        tp1_dict = pickle.load(f)

    with open(tp2_filename, "rb") as f:
        tp2_dict = pickle.load(f)

    # Read exogenous series dimension and data values
    # Check dimension values consistency between the two parts
    y_list = tp1_dict['latitude']
    assert np.all(np.array(y_list) == np.array(tp2_dict['latitude']))
    x_list = tp1_dict['longitude']
    assert np.all(np.array(x_list) == np.array(tp2_dict['longitude']))

    times1 = tp1_dict['time']
    matrix1 = np.array(tp1_dict['tp'])
    times2 = tp2_dict['time']
    matrix2 = np.array(tp2_dict['tp'])

    # Check part 1 is before of part 2
    assert times1[-1] < times2[0]
    times = times1 + times2

    # For each x and y crate time-series DataFrame
    tp_dict = {}
    # for i, y in tqdm(enumerate(y_list)):  # Iterate over latitude (y)
    for i, y in enumerate(y_list):  # Iterate over latitude (y)
        for j, x in enumerate(x_list):  # Iterate over longitude (x)
            # Read values with coords x, y from part 1 and 2
            vals1 = matrix1[:, i, j]
            vals2 = matrix2[:, i, j]

            # Concat the two matrix
            vals = np.concatenate([vals1, vals2])
            # Create a unique time-series DataFrame
            vals = pd.DataFrame({'tp': vals}, index=times)
            vals = vals.sort_index(ascending=True)

            # Sampling based on freq
            vals = compress_ts(vals, method=tp_sum, freq='D')

            tp_dict[(x, y)] = vals
    return tp_dict


def unwrap_t2m_file(t2m_filename: str) -> Dict[Tuple[float, float], pd.DataFrame]:
    with open(t2m_filename, "rb") as f:
        t2m_dict = pickle.load(f)

    # Read exogenous series dimension and data values
    y_list = t2m_dict['latitude']
    x_list = t2m_dict['longitude']
    times = t2m_dict['time']
    matrix = np.array(t2m_dict['t2m'])

    # For each x and y crate time-series DataFrame
    t2m_dict = {}
    # for i, y in tqdm(enumerate(y_list)):  # Iterate over latitude (y)
    for i, y in enumerate(y_list):  # Iterate over latitude (y)
        for j, x in enumerate(x_list):  # Iterate over longitude (x)
            # Read values with coords x, y
            vals = matrix[:, i, j]

            # Create a unique time-series DataFrame
            vals = pd.DataFrame({'t2m': vals}, index=times)
            vals = vals.sort_index(ascending=True)

            # Sampling based on freq
            vals = compress_ts(vals, method=t2m_stats, freq='D')

            t2m_dict[(x, y)] = vals
    return t2m_dict


def merge_exogenous_dict(
        exg1: Dict[Tuple[float, float], pd.DataFrame],
        exg2: Dict[Tuple[float, float], pd.DataFrame],
) -> Dict[Tuple[float, float], pd.DataFrame]:
    assert exg1.keys() == exg2.keys()

    exg = {}
    for k, df1 in exg1.items():
        df2 = exg2[k]

        df = pd.concat([df1, df2], axis=1, join='inner')
        df = df.sort_index(ascending=True)
        exg[k] = df

    return exg


def create_exogenous_series():
    base_dir = '../../../../../Dataset/AdbPo/Piezo/'
    t2m_filename = os.path.join(base_dir, 'NetCDF', 't2m.pickle')  # temperature exogenous series
    tp1_filename = os.path.join(base_dir, 'NetCDF', 'tp_1.pickle')  # total precipitation variable
    tp2_filename = os.path.join(base_dir, 'NetCDF', 'tp_2.pickle')  # total precipitation variable

    # TP exogenous series
    tp_dict = combine_tp_files(tp1_filename, tp2_filename)
    out_filename = os.path.join(base_dir, 'NetCDF', 'exg_tp.pickle')
    with open(out_filename, 'wb') as f:
        pickle.dump(tp_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # T2M exogenous series
    t2m_dict = unwrap_t2m_file(t2m_filename)
    out_filename = os.path.join(base_dir, 'NetCDF', 'exg_t2m.pickle')
    with open(out_filename, 'wb') as f:
        pickle.dump(t2m_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Merge series
    exg_dict = merge_exogenous_dict(tp_dict, t2m_dict)

    out_filename = os.path.join(base_dir, 'NetCDF', 'exg_tp_t2m.pickle')
    with open(out_filename, 'wb') as f:
        pickle.dump(exg_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return

def compress_exogenous_series():
    def my_stats(df: pd.DataFrame) -> pd.Series:
        res = pd.Series({
            'tp': df['tp'].sum(),
            't2m_min': df['t2m_min'].min(),
            't2m_max': df['t2m_max'].max(),
            't2m_avg': df['t2m_avg'].mean()
        })
        res['__DATE__'] = df['__DATE__'].iloc[-1]
        return res

    base_dir = '../../../../../Dataset/AdbPo/Piezo/'
    filename = os.path.join(base_dir, 'NetCDF', 'exg_tp_t2m.pickle')
    out_filename = os.path.join(base_dir, 'NetCDF', 'exg_w_tp_t2m.pickle')
    with open(filename, 'rb') as f:
        exg_dict = pickle.load(f)

    # for k, ts in tqdm(exg_dict.items()):
    for k, ts in exg_dict.items():
        # Sampling based on freq
        ts = compress_ts(ts, method=my_stats, freq='W')
        exg_dict[k] = ts

    with open(out_filename, 'wb') as f:
        pickle.dump(exg_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # create_exogenous_series()
    # compress_exogenous_series()
    print('Hello World!')
