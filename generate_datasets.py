import argparse
import multiprocessing
import os
import json
import pickle
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

from ists.dataset.read import load_data
from ists.preparation import prepare_data, prepare_train_test, filter_data
from ists.preparation import define_feature_mask, get_list_null_max_size
from ists.preprocessing import get_time_max_sizes
from ists.spatial import prepare_exogenous_data, prepare_spatial_data

parser = argparse.ArgumentParser('FDB Dataset Generator')

parser.add_argument('--config', type=str, nargs='+', default='all',
                    help="Select different configs to generate the datasets.")

args = parser.parse_args()


def data_step(path_params: dict, prep_params: dict, eval_params: dict, keep_nan: bool = False,
              dataset_save_path: str = None) -> dict:
    ts_params = prep_params['ts_params']
    feat_params = prep_params['feat_params']
    spt_params = prep_params['spt_params']
    exg_params = prep_params['exg_params']

    # Load dataset
    ts_dict, exg_dict, spt_dict = load_data(
        ts_filename=path_params['ts_filename'],
        context_filename=path_params['ctx_filename'],
        ex_filename=path_params['ex_filename'],
        data_type=path_params['type'],
        nan_percentage=path_params['nan_percentage']
    )

    # Prepare x, y, time, dist, id matrix
    x_array, y_array, time_array, dist_x_array, dist_y_array, id_array = prepare_data(
        ts_dict=ts_dict,
        num_past=ts_params['num_past'],
        num_fut=ts_params['num_fut'],
        features=ts_params['features'],
        label_col=ts_params['label_col'],
        freq=ts_params['freq'],
        null_feat=feat_params['null_feat'],
        null_max_dist=feat_params['null_max_dist'],
        time_feats=feat_params['time_feats'],
        with_fill=not keep_nan
    )
    # print(f'Num of records raw: {len(x_array)}')
    # Compute feature mask and time encoding max sizes
    x_feature_mask = define_feature_mask(
        base_features=ts_params['features'],
        null_feat=feat_params['null_feat'],
        time_feats=feat_params['time_feats']
    )
    x_time_max_sizes = get_time_max_sizes(feat_params['time_feats'])
    # print(f'Feature mask: {x_feature_mask}')

    # Prepare spatial matrix
    spt_array, mask = prepare_spatial_data(
        x_array=x_array,
        id_array=id_array,
        time_array=time_array[:, 1],
        dist_x_array=dist_x_array,
        num_past=spt_params['num_past'],
        num_spt=spt_params['num_spt'],
        spt_dict=spt_dict,
        max_dist_th=spt_params['max_dist_th'],
        max_null_th=spt_params['max_null_th']
    )
    x_array = x_array[mask]
    y_array = y_array[mask]
    time_array = time_array[mask]
    dist_x_array = dist_x_array[mask]
    dist_y_array = dist_y_array[mask]
    id_array = id_array[mask]
    # print(f'Num of records after spatial augmentation: {len(x_array)}')

    # Filter data before
    x_array, y_array, time_array, dist_x_array, dist_y_array, id_array, spt_array = filter_data(
        x_array=x_array,
        y_array=y_array,
        time_array=time_array,
        dist_x_array=dist_x_array,
        dist_y_array=dist_y_array,
        id_array=id_array,
        spt_array=spt_array,
        train_start=eval_params['train_start'],
        max_label_th=eval_params['label_th'],
        max_null_th=eval_params['null_th']
    )
    # print(f'Num of records after null filter: {len(x_array)}')

    # Prepare exogenous matrix
    exg_array, mask = prepare_exogenous_data(
        id_array=id_array,
        time_array=time_array[:, 1],
        exg_dict=exg_dict,
        num_past=exg_params['num_past'],
        features=exg_params['features'],
        time_feats=exg_params['time_feats']
    )
    # Compute exogenous feature mask and time encoding max sizes
    exg_feature_mask = define_feature_mask(base_features=exg_params['features'], time_feats=exg_params['time_feats'])
    exg_time_max_sizes = get_time_max_sizes(exg_params['time_feats'])

    x_array = x_array[mask]
    y_array = y_array[mask]
    time_array = time_array[mask]
    dist_x_array = dist_x_array[mask]
    dist_y_array = dist_y_array[mask]
    id_array = id_array[mask]
    spt_array = [arr[mask] for arr in spt_array]
    # print(f'Num of records after exogenous augmentation: {len(x_array)}')

    res = prepare_train_test(
        x_array=x_array,
        y_array=y_array,
        time_array=time_array,
        dist_x_array=dist_x_array,
        dist_y_array=dist_y_array,
        id_array=id_array,
        spt_array=spt_array,
        exg_array=exg_array,
        test_start=eval_params['test_start'],
    )
    # print(f"X train: {len(res['x_train'])}")
    # print(f"X test: {len(res['x_test'])}")

    # Save extra params in train test dictionary
    # Save x and exogenous array feature mask
    res['x_feat_mask'] = x_feature_mask
    res['exg_feat_mask'] = exg_feature_mask

    # Save null max size by finding the maximum between train and test if any
    res['null_max_size'] = get_list_null_max_size(
        [res['x_train']] + [res['x_test']] + res['spt_train'] + res['spt_test'],
        x_feature_mask
    )

    # Save time max sizes
    res['time_max_sizes'] = x_time_max_sizes
    res['exg_time_max_sizes'] = exg_time_max_sizes

    if dataset_save_path:
        with open(dataset_save_path, "wb") as f:
            pickle.dump(res, f)
            print(f"Dataset {dataset_save_path} saved.")
    else:
        return res


def main():
    conf_files = {  # config name: data folder
        'config/params_french.json': 'data/pickles/french',
        'config/params_french_baseline.json': 'data/pickles/french_baseline',
        'config/params_ushcn.json': 'data/pickles/ushcn',
        'config/params_ushcn_baseline.json': 'data/pickles/ushcn_baseline'
    }

    subsets = [
        'subset_agg_th1_0.csv',
        'subset_agg_th1_1.csv',
        'subset_agg_th1_2.csv',
        'subset_agg_th15_0.csv',
        'subset_agg_th15_1.csv',
        'subset_agg_th15_2.csv',
        'subset_agg_th15_3.csv'
    ]

    nan_nums = [
        0.0,
        0.2,
        0.5,
        0.8
    ]

    num_futs = [
        7,
        14,
        30,
        60
    ]

    os.makedirs('./data/pickles', exist_ok=True)
    for folder_path in conf_files.values():
        os.makedirs(folder_path, exist_ok=True)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for conf_file, dataset_folder in conf_files.items():
            if conf_file not in args.config:
                continue

            with open(conf_file, 'r') as f:
                conf = json.load(f)

            path_params, prep_params, eval_params, _ = conf['path_params'], conf['prep_params'], conf[
                'eval_params'], conf['model_params']

            for nan_num in nan_nums:
                for subset in subsets:
                    for num_fut in num_futs:
                        path_params["nan_percentage"] = nan_num
                        prep_params['ts_params']['num_fut'] = num_fut

                        dataset_name = (f"{path_params['type']}_"
                                        f"{subset.replace('subset_agg_', '').replace('.csv', '')}_"
                                        f"nan{int(nan_num * 10)}_nf{num_fut}.pickle")

                        if 'ushcn' in dataset_name:
                            path_params["ex_filename"] = "./data/USHCN/" + subset
                        else:
                            path_params['ex_filename'] = "./data/FrenchPiezo/" + subset

                        print(f"Starting dataset {dataset_name} generation.")
                        futures_ = list()
                        dataset_complete_path = os.path.join(dataset_folder, dataset_name)
                        futures_.append(executor.submit(data_step,
                                                        deepcopy(path_params),
                                                        deepcopy(prep_params),
                                                        deepcopy(eval_params),
                                                        **{'keep_nan': False,
                                                           'dataset_save_path': dataset_complete_path
                                                           }))

        for process in futures.as_completed(futures_):
            if process.exception():
                print("WARNING: Exception during the generation of the datasets: ", process.exception())


if __name__ == '__main__':
    main()
