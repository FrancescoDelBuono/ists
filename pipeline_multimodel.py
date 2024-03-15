import multiprocessing
import os
import json
import argparse
import pickle
import shutil
import socket
import sys
import traceback
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from multiprocessing import Manager, get_context

import pandas as pd

from ablation import ablation, ablation_tests_mapping, single_test_ablation
from ists.dataset.read import load_data
from ists.preparation import prepare_data, prepare_train_test, filter_data
from ists.preparation import define_feature_mask, get_list_null_max_size
from ists.preprocessing import get_time_max_sizes
from ists.spatial import prepare_exogenous_data, prepare_spatial_data
from ists.model.wrapper import ModelWrapper
from ists.metrics import compute_metrics

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default=None,
                    help='the path where the configuration is stored.')
parser.add_argument('--device', type=str, nargs='+',
                    default=['cuda:0'] if len(tf.config.list_physical_devices('GPU')) > 0 else ['cpu'],
                    help='device to use (cpu, cuda:0, cuda:1, ...)')

parser.add_argument('--model', type=str, nargs='+', default=['ISTS'],
                    help='model to use (ISTS, CRU, mTAN, GRU-D, ...)')

parser.add_argument('--num_fut', type=int, nargs='+', default=[7, 14, 30, 60],
                    help='number of days from the present to predict.')
parser.add_argument('--nan_num', type=float, nargs='+', default=[0.0, 0.2, 0.5, 0.8])
parser.add_argument('--subset', type=str, nargs='+',
                    default=['subset_agg_th1_0.csv', 'subset_agg_th1_1.csv', 'subset_agg_th1_2.csv',
                             'subset_agg_th15_0.csv', 'subset_agg_th15_1.csv', 'subset_agg_th15_2.csv',
                             'subset_agg_th15_3.csv'
                             ],
                    help='subset of the dataset to use.')
parser.add_argument('--model_type', type=str, nargs='+',
                    default=['sttransformer', 't', 's', 'e', 'ts', 'te', 'se', 'stt_no_embd'])

parser.add_argument('--batch_run', action='store_true', default=False,
                    help='Run the models in parallel on multiple datasets and GPUs.')

parser.add_argument('--ablation_run', action='store_true', default=False,
                    help='Run the ablation experiments in parallel on multiple GPUs, in batch.')

parser.add_argument('--single_test_ablation', action='store_true', default=False,
                    help='Run the ablation experiments in parallel on multiple GPUs, one by one.')

parser.add_argument('--dataset_folder', type=str, default=None,
                    help='Folder containing the datasets for batch run.')

parser.add_argument('--recycle_gpu', action='store_true', default=False,
                    help='Use the same GPU for multiple processes, exploiting the available VRAM.')

parser.add_argument('--encoder_ablation', action='store_true', default=False,
                    help='Run encoder ablation experiments for ISTS.')

parser.add_argument('--embedder_ablation', action='store_true', default=False,
                    help='Run embedder ablation experiments for ISTS.')

parser.add_argument('--ablation_experiments', nargs='+', type=int, default=[1, 2, 3, 4],
                    help='Experiment numbers to run for the embedder ablation, from 1 to 4.')

parser.add_argument('--merge_only', action='store_true', default=False,
                    help='Only merge the results of the ablation tests, without running them.')

parser.add_argument('--force_execution', action='store_true', default=False,
                    help='Force the execution of the baseline models, even if the results already exist.')

parser.add_argument('--scaler', type=str, nargs='+', default=['standard'] * 3,
                    help='Scaler to use for the baseline models.')

args = parser.parse_args()

if args.scaler[0].lower() == 'none':
    args.scaler = None

embedder_ablation_experiments_map = {
    1: {
        "feat_params": {
            'null_feat': None,
            'null_max_dist': 12,
            'time_feats': []
        }
    },
    2: {
        "feat_params": {
            'null_feat': 'code_bool',
            'null_max_dist': 12,
            'time_feats': []
        }
    },
    3: {
        "feat_params": {
            'null_feat': None,
            'null_max_dist': 12,
            'time_feats': ['WY']
        }
    },
    4: {
        "model_params": {
            "model_type": "stt_no_embd",
            "nn_params": {
                "kernel_size": 5,
                "d_model": 64,
                "num_heads": 4,
                "dff": 128,
                "fff": 64,
                "activation": "relu",
                "exg_cnn": True,
                "spt_cnn": True,
                "time_cnn": True,
                "num_layers": 1,
                "with_cross": True,
                "dropout_rate": 0.1
            },
        },
        "feat_params": {
            "null_feat": None,
            "null_max_dist": 12,
            "time_feats": []
        },
        "exg_params": {
            "time_feats": []
        }
    }
}


def parse_params(config_file: str = None):
    """ Parse input parameters. """

    if args.device != ['all'] and args.device[0] != 'cpu':
        gpus = tf.config.list_physical_devices('GPU')
        gpus_idx = [int(dev[-1]) for dev in args.device]
        selected_gpus = [gpu for i, gpu in enumerate(gpus) if i in gpus_idx]
        tf.config.set_visible_devices(selected_gpus, 'GPU')
        for gpu in selected_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if config_file is None:
        config_file = args.file

    assert os.path.exists(config_file), 'Configuration file does not exist'

    with open(config_file, 'r') as f:
        conf = json.load(f)

    return conf['path_params'], conf['prep_params'], conf['eval_params'], conf['model_params']


def change_params(path_params: dict, base_string: str, new_string: str) -> dict:
    path_params['ts_filename'] = path_params['ts_filename'].replace(base_string, new_string, 1)
    path_params['ctx_filename'] = path_params['ctx_filename'].replace(base_string, new_string, 1)
    path_params['ex_filename'] = path_params['ex_filename'].replace(base_string, new_string, 1)

    return path_params


def data_step(path_params: dict, prep_params: dict, eval_params: dict, keep_nan: bool = False) -> dict:
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
    print(f'Num of records raw: {len(x_array)}')
    # Compute feature mask and time encoding max sizes
    x_feature_mask = define_feature_mask(
        base_features=ts_params['features'],
        null_feat=feat_params['null_feat'],
        time_feats=feat_params['time_feats']
    )
    x_time_max_sizes = get_time_max_sizes(feat_params['time_feats'])
    print(f'Feature mask: {x_feature_mask}')

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
    print(f'Num of records after spatial augmentation: {len(x_array)}')

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
    print(f'Num of records after null filter: {len(x_array)}')

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
    print(f'Num of records after exogenous augmentation: {len(x_array)}')

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
    print(f"X train: {len(res['x_train'])}")
    print(f"X test: {len(res['x_test'])}")

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

    return res


def model_step(train_test_dict: dict, model_params: dict, checkpoint_dir: str) -> dict:
    model_type = model_params['model_type']
    transform_type = model_params['transform_type']
    nn_params = model_params['nn_params']
    loss = model_params['loss']
    lr = model_params['lr']
    epochs = model_params['epochs']
    batch_size = model_params['batch_size']

    # Insert data params in nn_params for building the correct model
    nn_params['feature_mask'] = train_test_dict['x_feat_mask']
    nn_params['exg_feature_mask'] = train_test_dict['exg_feat_mask']
    nn_params['spatial_size'] = len(train_test_dict['spt_train'])
    nn_params['null_max_size'] = train_test_dict['null_max_size']
    nn_params['time_max_sizes'] = train_test_dict['time_max_sizes']
    nn_params['exg_time_max_sizes'] = train_test_dict['exg_time_max_sizes']

    model = ModelWrapper(
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
        model_params=nn_params,
        transform_type=transform_type,
        loss=loss,
        lr=lr,
    )

    model.fit(
        x=train_test_dict['x_train'],
        spt=train_test_dict['spt_train'],
        exg=train_test_dict['exg_train'],
        y=train_test_dict['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
        extra={
            'x': train_test_dict['x_test'],
            'spt': train_test_dict['spt_test'],
            'exg': train_test_dict['exg_test'],
            'y': train_test_dict['y_test']
        }
    )

    preds = model.predict(
        x=train_test_dict['x_test'],
        spt=train_test_dict['spt_test'],
        exg=train_test_dict['exg_test'],
    )

    res = {}
    res_test = compute_metrics(y_true=train_test_dict['y_test'], y_preds=preds)
    res_test = {f'test_{k}': val for k, val in res_test.items()}
    res.update(res_test)

    preds = model.predict(
        x=train_test_dict['x_train'],
        spt=train_test_dict['spt_train'],
        exg=train_test_dict['exg_train'],
    )
    res_train = compute_metrics(y_true=train_test_dict['y_train'], y_preds=preds)
    res_train = {f'train_{k}': val for k, val in res_train.items()}
    res.update(res_train)

    res['loss'] = model.history.history['loss']
    res['val_loss'] = model.history.history['val_loss']
    print(res_test)
    return res


def batch_run():
    models = args.model
    dataset_folder = args.dataset_folder

    if args.model == ['ISTS']:
        for dataset in dataset_folder.listdir():
            print(f"Dataset: {dataset}")

            if dataset.startswith('ushcn'):
                _, _, _, model_params = parse_params('config/params_ushcn.json')
            elif dataset.startswith('french'):
                _, _, _, model_params = parse_params('config/params_french.json')
            else:
                raise ValueError('Unsupported dataset.')

            checkpoint_dir = f'./output_{dataset.split(".pickle")[0]}'

            with open(os.path.join(os.getcwd(), dataset_folder, dataset), 'rb') as dataset_file:
                train_test_dict = pickle.load(dataset_file)

            model_step(train_test_dict, model_params, checkpoint_dir)

    else:  # CRU, mTAN, GRU-D
        if args.device == ['all']:
            devices = 'all'

        elif isinstance(args.device, list) and len(args.device) > 1:
            devices = ' '.join(args.device)

        else:
            devices = args.device[0]

        python_interpreter_path = os.path.join(os.path.expanduser('~'),
                                               '.virtualenvs', 'ists', 'bin', 'python')
        command = (f'{python_interpreter_path} launch_experiments.py '
                   f'--model {" ".join(models)} '
                   f'--dataset {os.path.abspath(dataset_folder)} '
                   f'--device {devices} '
                   f'{"--recycle_gpu" if args.recycle_gpu else ""} '
                   f'{"--force_execution" if args.force_execution else ""} ')

        if args.scaler:
            if len(args.scaler) > 1:
                command += f'--scaler {" ".join(args.scaler)}'
            else:
                command += f'--scaler {args.scaler[0]}'

        else:
            command += '--scaler None'

        print(command)

        os.system(command)

    print("Batch run ended.")


def run_parallel_ablation(gpu_locks: list, path_params: dict, prep_params: dict, eval_params: dict,
                          model_params: dict, res_dir: str, data_dir: str, model_dir: str):
    device_idx = set_computing_device(gpu_locks)

    try:
        ablation(
            path_params=path_params,
            prep_params=prep_params,
            eval_params=eval_params,
            model_params=model_params,
            res_dir=res_dir,
            data_dir=data_dir,
            model_dir=model_dir,
            ablation_embedder=True,
            ablation_encoder=True,
        )

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise e

    finally:  # this is always executed if an exception is raised or not
        gpu_locks[device_idx].release()


def run_parallel_single_test_ablation(gpu_locks: list[multiprocessing.RLock],
                                      dataset_locks: dict[str: multiprocessing.RLock],
                                      ablation_test: str,
                                      path_params: dict, prep_params: dict, eval_params: dict,
                                      model_params: dict, res_dir: str, data_dir: str, model_dir: str):
    device_idx = set_computing_device(gpu_locks)

    subset = (os.path.basename(path_params['ex_filename'])
              .replace('subset_agg_', '')
              .replace('.csv', ''))

    nan_percentage = path_params['nan_percentage']
    num_fut = prep_params['ts_params']['num_fut']

    os.makedirs(data_dir, exist_ok=True)

    out_name = f"{path_params['type']}_{subset}_nan{int(nan_percentage * 10)}_nf{num_fut}"
    dataset_name = f"{out_name}.pickle"
    pickle_path = os.path.join(data_dir, dataset_name)

    with dataset_locks[dataset_name]:
        if not os.path.exists(pickle_path):
            print("Process", os.getpid(), "is preparing dataset", dataset_name, flush=True)
            train_test_dict = data_step(path_params, prep_params, eval_params, keep_nan=False)

            with open(pickle_path, "wb") as f:
                train_test_dict['params'] = {
                    'path_params': path_params,
                    'prep_params': prep_params,
                    'eval_params': eval_params,
                    'model_params': model_params,
                }
                pickle.dump(train_test_dict, f)

            del train_test_dict

    print(f"{os.getpid()}: Invoking single test ablation with test {ablation_test}.")

    try:
        single_test_ablation(path_params=path_params, prep_params=prep_params, eval_params=eval_params,
                             model_params=model_params, res_dir=res_dir, data_dir=data_dir, model_dir=model_dir,
                             ablation_test=ablation_test)

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise e

    finally:  # this is always executed if an exception is raised or not
        gpu_locks[device_idx].release()
        checkpoint_path = os.path.join(model_dir, f"{out_name}_{os.getpid()}")
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)


def set_computing_device(gpu_locks):
    device = 'cpu'
    device_idx = -1
    if args.device == ['all']:
        args.device = ['cuda:{}'.format(i) for i in range(len(tf.config.list_physical_devices('GPU')))]
    for i, lock in enumerate(gpu_locks):
        if lock.acquire(blocking=False):
            device = args.device[i]
            device_idx = i
            break
    print(f"{os.getpid()}: Running on device {device}")
    gpus = tf.config.list_physical_devices('GPU')
    gpu_idx = int(device[-1])
    try:
        tf.config.set_visible_devices(gpus[gpu_idx], 'GPU')
    except RuntimeError as re:
        print(f"Could not set visible devices on device {device}.")
        print(re)
        raise re

    # TODO: this could cause errors if the device is already initialized for some reason. Investigate.
    try:
        tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
    except RuntimeError as re:
        print(f"Could not set memory growth on device {device}.")
        print(re)
        print("Continuing.")

    return device_idx


def schedule_single_test_ablation(path_params: dict, prep_params: dict, eval_params: dict, model_params: dict,
                                  res_dir: str, data_dir: str, model_dir: str):
    subset = (os.path.basename(path_params['ex_filename'])
              .replace('subset_agg_', '')
              .replace('.csv', ''))

    results_filenames = [f"{path_params['type']}_{subset}_nan{int(nan_num * 10)}_nf{num_fut}"
                         for nan_num in args.nan_num for num_fut in args.num_fut]

    with Manager() as manager:
        gpu_locks = manager.list([manager.RLock() for _ in range(len(args.device))])
        dataset_locks = manager.dict({results_filename + '.pickle': manager.RLock()
                                      for results_filename in results_filenames})

        four_hours = 14400  # seconds

        for nan_num in args.nan_num:
            for num_fut in args.num_fut:
                # initialize the ProcessPool each time as already initialized devices cannot be modified
                # in TensorFlow. This joins the processes and spawns new ones, avoiding the problem.
                with ProcessPoolExecutor(max_workers=len(args.device),
                                         mp_context=get_context('spawn')) as executor:
                    futures_ = list()
                    # print(f"nan_num: {nan_num}")
                    # print(f"num_fut: {num_fut}")

                    path_params["nan_percentage"] = nan_num
                    prep_params['ts_params']['num_fut'] = num_fut

                    results_filename = (f"{path_params['type']}_{subset}"
                                        f"_nan{int(nan_num * 10)}_nf{num_fut}.csv")
                    dataset_filename = (f"{path_params['type']}_{subset}"
                                        f"_nan{int(nan_num * 10)}_nf{num_fut}.pickle")

                    print("Launching single test ablation processes on dataset:", dataset_filename)
                    print("Saving results in:", results_filename)

                    for test in ablation_tests_mapping.keys():
                        print(f"Scheduling {test}")

                        futures_.append(executor.submit(run_parallel_single_test_ablation,
                                                        gpu_locks, dataset_locks, test,
                                                        path_params, prep_params, eval_params, model_params,
                                                        res_dir, data_dir, model_dir))

                done, not_done = futures.wait(futures_, return_when=futures.FIRST_EXCEPTION, timeout=four_hours)

                futures_exceptions = [future.exception() for future in done]
                failed_futures = sum(map(lambda exception_: True if exception_ is not None else False,
                                         futures_exceptions))

                if failed_futures > 0:
                    print("Could not run ablation tests. Thrown exceptions: ")

                    for exception in futures_exceptions:
                        print(exception)

                    print("traceback.print_exc()")
                    traceback.print_exc(file=sys.stderr)

                    raise RuntimeError(f"Could not run ablation tests, {failed_futures} processes failed.")


def ablation_run():
    res_dir = './output/results'
    data_dir = './output/pickle'
    model_dir = './output/model'

    if args.device == ['all']:
        args.device = ['cuda:{}'.format(i) for i in range(len(tf.config.list_physical_devices('GPU')))]

    path_params, prep_params, eval_params, model_params = parse_params()
    # path_params = change_params(path_params, '../../data', '../../Dataset/AdbPo')

    # if 'gnode01' in socket.gethostname():
    # print("Running on ARIES.")
    if args.single_test_ablation:
        results_filenames = list()
        results_files = dict()

        if args.subset[0] != 'None':
            for subset in args.subset:
                if 'ushcn' in args.file.lower():
                    path_params["ex_filename"] = "./data/USHCN/" + subset
                else:
                    path_params['ex_filename'] = "./data/FrenchPiezo/" + subset

                subset = (os.path.basename(path_params['ex_filename'])
                          .replace('subset_agg_', '')
                          .replace('.csv', ''))

                subset_results_filenames = [f"{path_params['type']}_{subset}_nan{int(nan_num * 10)}_nf{num_fut}"
                                            for nan_num in args.nan_num for num_fut in args.num_fut]

                subset_results_files = {results_filename + '.csv': ['_'.join([results_filename, ablation_test]) + '.csv'
                                                                    for ablation_test in ablation_tests_mapping.keys()]
                                        for results_filename in subset_results_filenames
                                        }

                results_filenames += subset_results_filenames
                results_files.update(subset_results_files)

                if not os.path.exists(path_params['ex_filename']):
                    raise FileNotFoundError(f"File {path_params['ex_filename']} does not exist.")

                if not args.merge_only:
                    schedule_single_test_ablation(path_params, prep_params, eval_params, model_params,
                                                  res_dir, data_dir, model_dir)

        else:
            if 'ushcn' in args.file.lower():
                # path_params["ex_filename"] = "./data/USHCN/" + subset
                path_params["ex_filename"] = "./data/USHCN/" + 'pivot_1990_1993_thr4_normalize.csv'
            else:
                path_params['ex_filename'] = "./data/FrenchPiezo/" + 'dataset_2015_2021.csv'

            if not args.merge_only:
                schedule_single_test_ablation(path_params, prep_params, eval_params, model_params,
                                              res_dir, data_dir, model_dir)

            subset = (os.path.basename(path_params['ex_filename'])
                      .replace('subset_agg_', '')
                      .replace('.csv', ''))

            results_filenames = [f"{path_params['type']}_{subset}_nan{int(nan_num * 10)}_nf{num_fut}"
                                 for nan_num in args.nan_num for num_fut in args.num_fut]

            results_files = {results_filename + '.csv': ['_'.join([results_filename, ablation_test]) + '.csv'
                                                         for ablation_test in ablation_tests_mapping.keys()]
                             for results_filename in results_filenames
                             }

        print("All ablation tests ran successfully. Merging results.")

        for results_filename, test_results_filenames in results_files.items():
            print("Merging results for", results_filename)
            print("res_dir", res_dir)
            print("results_filename", results_filename)
            complete_results_path = os.path.join(res_dir, results_filename)
            print("complete_results_path", complete_results_path)
            if os.path.exists(complete_results_path):
                results_df = pd.read_csv(complete_results_path, index_col=0)
            else:
                results_df = pd.DataFrame()
            for test_results_filename in test_results_filenames:
                print("test_results_filename", test_results_filename)
                test_results_path = os.path.join(res_dir, test_results_filename)
                print("test_results_path", test_results_path)

                if not os.path.exists(test_results_path):
                    print("WARNING: File", test_results_filename, "does not exist. Skipping.", file=sys.stderr)
                    continue
                test_results_df = pd.read_csv(test_results_path, index_col=0)
                results_df = pd.concat([results_df, test_results_df], axis=1)

            results_df.T.to_csv(os.path.join(res_dir, results_filename), index=True)

    else:
        if len(args.device) == 1:
            # if len(args.subset) > 1 or len(args.nan_num) > 1 or len(args.num_fut) > 1:
            if len(args.nan_num) > 1 or len(args.num_fut) > 1:
                print("Single device, multiple parameter sets.")

                # for subset in args.subset:
                for nan_num in args.nan_num:
                    for num_fut in args.num_fut:
                        print("Launching ablation with params:")
                        # print(f"subset: {subset}")
                        print(f"nan_num: {nan_num}")
                        print(f"num_fut: {num_fut}")

                        if 'ushcn' in args.file.lower():
                            # path_params["ex_filename"] = "./data/USHCN/" + subset
                            path_params["ex_filename"] = "./data/USHCN/" + 'pivot_1990_1993_thr4_normalize.csv'
                        else:
                            path_params['ex_filename'] = "./data/FrenchPiezo/" + 'dataset_2015_2021.csv'

                        path_params["nan_percentage"] = nan_num
                        prep_params['ts_params']['num_fut'] = num_fut

                        ablation(path_params, prep_params, eval_params, model_params, res_dir, data_dir, model_dir)

            else:
                print("Single device, single parameter set.")
                # subset = args.subset[0]
                nan_num = args.nan_num[0]
                num_fut = args.num_fut[0]

                if 'ushcn' in args.file.lower():
                    # path_params["ex_filename"] = "./data/USHCN/" + subset
                    path_params["ex_filename"] = "./data/USHCN/" + 'pivot_1990_1993_thr4_normalize.csv'
                else:
                    path_params['ex_filename'] = "./data/FrenchPiezo/" + 'dataset_2015_2021.csv'

                path_params["nan_percentage"] = nan_num
                prep_params['ts_params']['num_fut'] = num_fut

                # model_params['lr'] = 1e-3
                # model_params['epochs'] = 10

                ablation(path_params, prep_params, eval_params, model_params, res_dir, data_dir, model_dir)

        else:
            print(f"Running on {socket.gethostname()}.")
            with Manager() as manager:
                print("manager, args.device", args.device)
                gpu_locks = [manager.Lock() for _ in range(len(args.device))]
                with ProcessPoolExecutor(max_workers=len(args.device), mp_context=get_context('spawn')) as executor:
                    futures_ = list()
                    # for subset in args.subset:
                    for nan_num in args.nan_num:
                        for num_fut in args.num_fut:
                            print("Launching ablation with params:")
                            # print(f"subset: {subset}")
                            print(f"nan_num: {nan_num}")
                            print(f"num_fut: {num_fut}")

                            if 'ushcn' in args.file.lower():
                                # path_params["ex_filename"] = "./data/USHCN/" + subset
                                path_params["ex_filename"] = "./data/USHCN/" + 'pivot_1990_1993_thr4_normalize.csv'
                            else:
                                path_params['ex_filename'] = "./data/FrenchPiezo/" + 'dataset_2015_2021.csv'

                            path_params["nan_percentage"] = nan_num
                            prep_params['ts_params']['num_fut'] = num_fut

                            futures_.append(executor.submit(run_parallel_ablation, gpu_locks,
                                                            path_params, prep_params, eval_params, model_params,
                                                            res_dir, data_dir, model_dir))

                    done, not_done = futures.wait(futures_, return_when=futures.FIRST_EXCEPTION)

                    futures_exceptions = [future.exception() for future in done]
                    failed_futures = sum(map(lambda exception_: True if exception_ is not None else False,
                                             futures_exceptions))

                    if failed_futures > 0:
                        print("Could not run ablation tests. Thrown exceptions: ")

                        for exception in futures_exceptions:
                            print(exception)

                        raise RuntimeError(f"Could not run ablation tests, {failed_futures} processes failed.")


def no_subset_run():
    path_params, prep_params, eval_params, model_params = parse_params()

    if 'ushcn' in args.file.lower():
        args.subset = ['pivot_1990_1993_thr4_normalize.csv']
    else:
        args.subset = ['dataset_2015_2021.csv']

    subset = args.subset[0]

    datasets_folder = os.path.join(os.getcwd(), 'output', 'pickle',
                                   'ushcn' if 'ushcn' in args.file.lower() else 'french')

    os.makedirs(datasets_folder, exist_ok=True)

    print("Generating datasets: ")
    for num_fut in args.num_fut:
        for nan_num in args.nan_num:
            dataset_name = (f"{path_params['type']}_"
                            f"{subset.replace('subset_agg_', '').replace('.csv', '')}_"
                            f"nan{int(nan_num * 10)}_"
                            f"nf{num_fut}")

            print(f"Dataset: {dataset_name}")

            path_params["nan_percentage"] = nan_num
            prep_params['ts_params']['num_fut'] = num_fut

            if dataset_name + '.pickle' in os.listdir(datasets_folder):
                print(f"Dataset {dataset_name} already exists. Skipping.")
                continue

            train_test_dict = data_step(path_params, prep_params, eval_params, keep_nan=True)

            with open(os.path.join(datasets_folder, f'{dataset_name}.pickle'), 'wb') as f:
                pickle.dump(train_test_dict, f)

    print(f"Datasets generated. Launching batch run on {', '.join(args.model)} using devices {', '.join(args.device)}.")
    args.dataset_folder = datasets_folder
    batch_run()


def normal_run():
    path_params, prep_params, eval_params, model_params = parse_params()

    if args.file is None:
        raise ValueError("Config file is mandatory if --batch_run is not specified.")

    if (('ISTS' in args.model and 'baseline' in args.file) or
            ('ISTS' not in args.model and 'baseline' not in args.file)):
        raise ValueError('Baseline config files are for models different from ISTS.')

    tmp_datasets_path = str()

    if args.model != ['ISTS']:
        tmp_datasets_path = f'./tmp_datasets_{os.getpid()}'
        os.makedirs(tmp_datasets_path, exist_ok=True)

    run_num = 0

    for num_fut in args.num_fut:
        for nan_num in args.nan_num:
            for subset in args.subset:
                dataset_name = (f"{path_params['type']}_"
                                f"{subset.replace('subset_agg_', '').replace('.csv', '')}_"
                                f"nan{int(nan_num * 10)}_"
                                f"nf{num_fut}")

                models = args.model.copy()
                run_num += 1

                print(f"Run number: {run_num}")
                print(f"Models: {models}")
                print(f"Nan percentage: {nan_num}")
                print(f"dataset: {dataset_name}")
                print(f"num_fut: {num_fut}")

                if 'ushcn' in dataset_name:
                    path_params["ex_filename"] = "./data/USHCN/" + subset
                else:
                    path_params['ex_filename'] = "./data/FrenchPiezo/" + subset

                path_params["nan_percentage"] = nan_num
                prep_params['ts_params']['num_fut'] = num_fut

                try:
                    if 'ISTS' in models:
                        if args.embedder_ablation:
                            model_type = 'sttransformer'
                            model_params['model_type'] = model_type
                            for experiment in args.ablation_experiments:
                                print(f"Running embedder ablation experiment {experiment}.")
                                print(embedder_ablation_experiments_map[experiment])
                                print(model_params)
                                print(prep_params)
                                for key, params in embedder_ablation_experiments_map[experiment].items():
                                    if key == 'model_params':
                                        model_type = params['model_type']
                                        model_params.update(params)
                                    elif key in ('feat_params', 'exg_params'):
                                        prep_params[key].update(params)

                                checkpoint_dir = f'./output_{dataset_name}_{subset}_{nan_num}_{num_fut}_{model_type}'
                                train_test_dict = data_step(path_params, prep_params, eval_params, keep_nan=False)

                                start_time = datetime.now()
                                results = model_step(train_test_dict, model_params, checkpoint_dir)
                                end_time = datetime.now()
                                print('Duration: {}'.format(end_time - start_time))
                                print(results)

                        elif args.encoder_ablation:
                            train_test_dict = data_step(path_params, prep_params, eval_params, keep_nan=False)

                            for model_type in args.model_type:
                                print(f"Running encoder ablation experiment with model {model_type}.")
                                model_params['model_type'] = model_type
                                checkpoint_dir = f'./output_{dataset_name}_{subset}_{nan_num}_{num_fut}_{model_type}'

                                start_time = datetime.now()
                                results = model_step(train_test_dict, model_params, checkpoint_dir)
                                end_time = datetime.now()
                                print('Duration: {}'.format(end_time - start_time))
                                print(results)

                        else:
                            model_type = 'sttransformer'
                            train_test_dict = data_step(path_params, prep_params, eval_params, keep_nan=False)
                            checkpoint_dir = f'./output_{dataset_name}_{subset}_{nan_num}_{num_fut}_{model_type}'

                            start_time = datetime.now()
                            results = model_step(train_test_dict, model_params, checkpoint_dir)
                            end_time = datetime.now()
                            print('Duration: {}'.format(end_time - start_time))
                            print(results)

                    else:  # CRU, mTAN, GRU-D
                        train_test_dict = data_step(path_params, prep_params, eval_params, keep_nan=True)

                        dataset_file_path = os.path.join(tmp_datasets_path, f'{dataset_name}.pickle')
                        with open(dataset_file_path, 'wb') as f:
                            pickle.dump(train_test_dict, f)

                        python_interpreter_path = os.path.join(os.path.expanduser('~'),
                                                               '.virtualenvs', 'ists', 'bin', 'python')
                        command = (f'{python_interpreter_path} launch_experiments.py '
                                   f'--model {" ".join(models)} '
                                   f'--dataset {os.path.abspath(dataset_file_path)} '
                                   f'--device {" ".join(args.device)} '
                                   f'{"--force_execution" if args.force_execution else ""} ')

                        if args.scaler:
                            if len(args.scaler) > 1:
                                command += f'--scaler {" ".join(args.scaler)}'
                            else:
                                command += f'--scaler {args.scaler[0]}'
                        else:
                            command += '--scaler None'

                        print(command)
                        try:
                            os.system(command)
                        except Exception as e:
                            print(f"Dataset {dataset_name} failed.")
                            raise e

                        finally:
                            os.system(f"rm {os.path.join(tmp_datasets_path, dataset_name)}*")

                except Exception as e:
                    print(f"Dataset {dataset_name} failed: {e}")

    if os.path.exists(tmp_datasets_path):
        os.rmdir(tmp_datasets_path)


def main():
    if args.batch_run:
        batch_run()

    elif args.ablation_run or args.single_test_ablation:
        ablation_run()

    elif args.subset == ['None']:
        no_subset_run()

    else:
        normal_run()


if __name__ == '__main__':
    main()
