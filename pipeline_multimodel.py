import os
import json
import argparse
import pickle

from ists.dataset.read import load_data
from ists.preparation import prepare_data, prepare_train_test, filter_data
from ists.preparation import define_feature_mask, get_list_null_max_size
from ists.preprocessing import get_time_max_sizes
from ists.spatial import prepare_exogenous_data, prepare_spatial_data
from ists.model.wrapper import ModelWrapper
from ists.metrics import compute_metrics

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True,
                    help='the path where the configuration is stored.')
parser.add_argument('--device', type=str, nargs='+',
                    default=['cuda:0'] if len(tf.config.list_physical_devices('GPU')) > 0 else ['cpu'],
                    help='device to use (cpu, cuda:0, cuda:1, ...)')

parser.add_argument('--model', type=str, nargs='+', default=['ISTS'],
                    help='model to use (ISTS, CRU, mTAN, GRU-D, ...)')

parser.add_argument('--num_fut', type=int, nargs='+', default=[7],
                    help='number of days from the present to predict.')
parser.add_argument('--nan_num', type=float, nargs='+', default=[0.0, 0.2, 0.5, 0.8])
parser.add_argument('--subset', type=str, nargs='+',
                    default=['subset_agg_th1_0.csv', 'subset_agg_th1_1.csv', 'subset_agg_th1_2.csv',
                             'subset_agg_th15_0.csv', 'subset_agg_th15_1.csv', 'subset_agg_th15_2.csv',
                             'subset_agg_th15_3.csv'
                             ],
                    help='subset of the dataset to use.')
parser.add_argument('--model_type', type=str, nargs='+',
                    default=['sttransformer', 't', 's', 'e', 'ts', 'te', 'se'])

args = parser.parse_args()


def parse_params():
    """ Parse input parameters. """

    if args.device[0] != 'cpu':
        gpus = tf.config.list_physical_devices('GPU')
        gpus_idx = [int(dev[-1]) for dev in args.device]
        selected_gpus = [gpu for i, gpu in enumerate(gpus) if i in gpus_idx]
        tf.config.set_visible_devices(selected_gpus, 'GPU')

    conf_file = args.file
    assert os.path.exists(conf_file), 'Configuration file does not exist'

    with open(conf_file, 'r') as f:
        conf = json.load(f)

    return conf['path_params'], conf['prep_params'], conf['eval_params'], conf['model_params']


def get_params():
    # Path params (i.e. time-series path, context table path)
    base_dir = '../../Dataset/AdbPo/Piezo/'
    path_params = {
        # main time-series (i.e. piezo time-series)
        'ts_filename': os.path.join(base_dir, 'ts_all.xlsx'),
        # table with context information (i.e. coordinates...)
        'ctx_filename': os.path.join(base_dir, 'data_ext_all.xlsx'),
        # dictionary of exogenous time-series (i.e. temperature...)
        'ex_filename': os.path.join(base_dir, 'NetCDF', 'exg_w_tp_t2m.pickle')
    }

    # Preprocessing params (i.e. num past, num future, sampling frequency, features....)
    prep_params = {
        'ts_params': {
            'features': ['Piezometria (m)'],
            'label_col': 'Piezometria (m)',
            'num_past': 48,
            'num_fut': 6,
            'freq': 'M',  # ['M', 'W', 'D']
        },
        'feat_params': {
            # Null Encoding
            'null_feat': 'code_lin',  # ['code_bool', 'code_lin', 'bool', 'lin', 'log']
            'null_max_dist': 12,
            # Time Encoding
            'time_feats': ['M']  # ['D', 'DW', 'WY', 'M']
        },
        'spt_params': {
            'num_past': 36,
            'num_spt': 5,
            'max_dist_th': 10000,
            'max_null_th': 13,
        },
        'exg_params': {
            'num_past': 72,
            'features': ['tp', 't2m_min', 't2m_max', 't2m_avg'],
            'time_feats': ['WY', 'M']
        },
    }

    # Evaluation train and test params
    eval_params = {
        'train_start': '2009-01-01',
        'test_start': '2019-01-01',
        'label_th': 1,
        'null_th': 13,
    }

    model_params = {
        'transform_type': 'standard',  # None 'minmax' 'standard'
        'model_type': 'sttransformer',  # 'sttransformer', 'dense', 'lstm', 'bilstm', 'lstm_base', 'bilstm_base'
        'nn_params': {
            'kernel_size': 3,
            'd_model': 32,
            'num_heads': 8,
            'dff': 64,
            'fff': 32,
            'activation': 'relu',
            'exg_cnn': True,
            'spt_cnn': True,
            'time_cnn': True,
            'dropout_rate': 0.2,
            'num_layers': 2,
            'with_cross': True,
        },
        'lr': 0.0,
        'loss': 'mse',
        'batch_size': 32,
        'epochs': 5
    }

    return path_params, prep_params, eval_params, model_params


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


def main():
    path_params, prep_params, eval_params, model_params = parse_params()

    if ('ISTS' in args.model and 'baseline' in args.file) or ('ISTS' not in args.model and 'baseline' not in args.file):
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
                                f"nan{int(nan_num * 10)}")

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
                        models.remove('ISTS')

                        print("Running ISTS first...")
                        train_test_dict = data_step(path_params, prep_params, eval_params,
                                                    keep_nan=False)
                        for model_type in args.model_type:
                            model_params['model_type'] = model_type
                            checkpoint_dir = f'./output_{dataset_name}_{subset}_{nan_num}_{num_fut}_{model_type}'
                            print(model_step(train_test_dict, model_params, checkpoint_dir))

                    if models:  # if there are other models to run, meaning the list is not empty
                        train_test_dict = data_step(path_params, prep_params, eval_params,
                                                    keep_nan=True)

                        dataset_file_path = os.path.join(tmp_datasets_path, f'{dataset_name}_nf{num_fut}.pickle')
                        with open(dataset_file_path, 'wb') as f:
                            pickle.dump(train_test_dict, f)

                        command = (f'python3 launch_experiments.py --model {" ".join(models)} --dataset '
                                   f'{os.path.abspath(dataset_file_path)} '
                                   f'--device {" ".join(args.device)}')

                        print(command)

                        os.system(command)

                        os.remove(dataset_file_path)

                except Exception as e:
                    print(f"Dataset {dataset_name} failed: {e}")

    if os.path.exists(tmp_datasets_path):
        os.rmdir(tmp_datasets_path)


if __name__ == '__main__':
    main()
