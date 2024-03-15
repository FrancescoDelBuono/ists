import multiprocessing
import os
import pickle
import shutil
import time

import numpy as np
import pandas as pd

from pipeline import data_step, model_step

def no_ablation(train_test_dict) -> dict:
    # train_test_dict['params']['model_params']['model_type'] = "sttransformer"
    return train_test_dict


def ablation_embedder_no_feat(train_test_dict, code) -> dict:
    for n in ['train', 'test']:
        cond_x = [x != code for x in train_test_dict['x_feat_mask']]
        train_test_dict[f'x_{n}'] = train_test_dict[f'x_{n}'][:, :, cond_x]
        train_test_dict[f'spt_{n}'] = [x[:, :, cond_x] for x in train_test_dict[f'spt_{n}']]

        cond_exg = [x != code for x in train_test_dict['exg_feat_mask']]
        train_test_dict[f'exg_{n}'] = train_test_dict[f'exg_{n}'][:, :, cond_exg]

    train_test_dict['x_feat_mask'] = [x for x in train_test_dict['x_feat_mask'] if x != code]
    train_test_dict['exg_feat_mask'] = [x for x in train_test_dict['exg_feat_mask'] if x != code]

    if code == 1:
        train_test_dict['null_max_size'] = None

    if code == 2:
        train_test_dict['time_max_sizes'] = []
        train_test_dict['exg_time_max_sizes'] = []

    return train_test_dict


def ablation_embedder_no_time(train_test_dict) -> dict:
    train_test_dict = ablation_embedder_no_feat(train_test_dict, 2)
    return train_test_dict


def ablation_embedder_no_null(train_test_dict) -> dict:
    train_test_dict = ablation_embedder_no_feat(train_test_dict, 1)
    return train_test_dict


def ablation_embedder_no_time_null(train_test_dict) -> dict:
    train_test_dict = ablation_embedder_no_feat(train_test_dict, 1)
    train_test_dict = ablation_embedder_no_feat(train_test_dict, 2)
    return train_test_dict


def ablation_encoder_stt(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "sttransformer"
    return train_test_dict


def ablation_encoder_t(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "t"
    return train_test_dict


def ablation_encoder_s(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "s"
    return train_test_dict


def ablation_encoder_e(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "e"
    return train_test_dict


def ablation_encoder_ts(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "ts"
    return train_test_dict


def ablation_encoder_te(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "te"
    return train_test_dict


def ablation_encoder_se(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "se"
    return train_test_dict


def ablation_encoder_ts_fe(train_test_dict) -> dict:
    # Models TS by concatenating exogenous features E in the feature dimension to T.
    train_test_dict['params']['model_params']['model_type'] = "ts_fe"
    return train_test_dict


def ablation_encoder_ts_fe_nonull(train_test_dict) -> dict:
    # Models TS by concatenating exogenous features E in the feature dimension to T,
    # without null encoding.
    train_test_dict = ablation_encoder_ts_fe(train_test_dict)
    train_test_dict = ablation_embedder_no_null(train_test_dict)
    return train_test_dict


def ablation_encoder_ts_fe_nonull_notime(train_test_dict) -> dict:
    # Models TS by concatenating exogenous features E in the feature dimension to T,
    # without null and time encoding.
    train_test_dict = ablation_encoder_ts_fe(train_test_dict)
    train_test_dict = ablation_embedder_no_time_null(train_test_dict)
    return train_test_dict


def ablation_encoder_stt_se(train_test_dict) -> dict:
    # Models STT by integrating exogenous E and T similarly to the S module.
    train_test_dict['params']['model_params']['model_type'] = "stt_se"
    return train_test_dict


def ablation_encoder_stt_se_nonull(train_test_dict) -> dict:
    # Models STT by integrating exogenous E and T similarly to the S module,
    # without null encoding.
    train_test_dict = ablation_encoder_stt_se(train_test_dict)
    train_test_dict = ablation_embedder_no_null(train_test_dict)
    return train_test_dict


def ablation_encoder_se_se(train_test_dict) -> dict:
    # Models SE by integrating exogenous E and T similarly to the S module.
    train_test_dict['params']['model_params']['model_type'] = "se_se"
    return train_test_dict


def ablation_encoder_se_se_nonull(train_test_dict) -> dict:
    # Models SE by integrating exogenous E and T similarly to the S module,
    # without null encoding.
    train_test_dict = ablation_encoder_se_se(train_test_dict)
    train_test_dict = ablation_embedder_no_null(train_test_dict)
    return train_test_dict


def ablation_encoder_stt_mts_e(train_test_dict) -> dict:
    # Models STT with multivariate inputs in E.
    cond_x = [x == 0 for x in train_test_dict['x_feat_mask']]
    for n in ['train', 'test']:
        x = train_test_dict[f'x_{n}'][:, :, cond_x].copy()

        train_test_dict[f'exg_{n}'] = np.concatenate([train_test_dict[f'exg_{n}'], x], axis=2)

    x_feat_mask = [x for x in train_test_dict['x_feat_mask'] if x == 0]
    train_test_dict['exg_feat_mask'] = train_test_dict['exg_feat_mask'] + x_feat_mask

    return train_test_dict


def ablation(
        path_params: dict,
        prep_params: dict,
        eval_params: dict,
        model_params: dict,
        res_dir: str,
        data_dir: str,
        model_dir: str,
        ablation_embedder: bool = True,
        ablation_encoder: bool = True,
        ablation_extra: dict = None
):
    subset = os.path.basename(path_params['ex_filename']).replace('subset_agg_', '').replace('.csv', '')
    nan_percentage = path_params['nan_percentage']
    num_fut = prep_params['ts_params']['num_fut']

    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    out_name = f"{path_params['type']}_{subset}_nan{int(nan_percentage * 10)}_nf{num_fut}"
    results_path = os.path.join(res_dir, f"{out_name}.csv")
    pickle_path = os.path.join(data_dir, f"{out_name}.pickle")
    checkpoint_path = os.path.join(model_dir, f"{out_name}")

    print("Saving results in ", results_path)
    print("Saving pickle in ", pickle_path)
    print("Saving model in ", checkpoint_path)

    results = {}

    train_test_dict = data_step(path_params, prep_params, eval_params, keep_nan=False)

    with open(pickle_path, "wb") as f:
        train_test_dict['params'] = {
            'path_params': path_params,
            'prep_params': prep_params,
            'eval_params': eval_params,
            'model_params': model_params,
        }
        pickle.dump(train_test_dict, f)

    selected_model = train_test_dict['params']['model_params']['model_type'][:3].upper()
    ablations_mapping = {
        selected_model: no_ablation,
    }

    if ablation_embedder:
        ablations_mapping.update({
            f'{selected_model} w/o time enc': ablation_embedder_no_time,
            f'{selected_model} w/o null enc': ablation_embedder_no_null,
            f'{selected_model} w/o time null enc': ablation_embedder_no_time_null,
        })

    if ablation_encoder:
        ablations_mapping.update({
            'T': ablation_encoder_t,
            'S': ablation_encoder_s,
            'E': ablation_encoder_e,
            'TE': ablation_encoder_te,
            'TS': ablation_encoder_ts,
            'SE': ablation_encoder_se,
        })

    ablation_extra_models = {
        'TS_FE': ablation_encoder_ts_fe,
        'STT_SE': ablation_encoder_stt_se,
        'SE_SE': ablation_encoder_se_se,
        'STT_MTS_E': ablation_encoder_stt_mts_e,
    }

    if ablation_extra:
        ablations_mapping.update(ablation_extra)
    else:
        ablations_mapping.update(ablation_extra_models)

    for name, func in ablations_mapping.items():
        # Load data
        with open(pickle_path, "rb") as f:
            train_test_dict = pickle.load(f)

        # Configure ablation test
        train_test_dict = func(train_test_dict)

        # Exec ablation test
        print(f"\n{name}: {train_test_dict['params']['model_params']['model_type']}")
        results[name] = model_step(train_test_dict, train_test_dict['params']['model_params'], checkpoint_path)

        # Save results
        pd.DataFrame(results).T.to_csv(results_path, index=True)

    return pd.DataFrame(results).T


ablation_tests_mapping = {
        'no_ablation': no_ablation,
        'ablation_embedder_no_time': ablation_embedder_no_time,
        'ablation_embedder_no_null': ablation_embedder_no_null,
        'ablation_embedder_no_time_null': ablation_embedder_no_time_null,
        'ablation_encoder_t': ablation_encoder_t,
        'ablation_encoder_s': ablation_encoder_s,
        'ablation_encoder_e': ablation_encoder_e,
        'ablation_encoder_te': ablation_encoder_te,
        'ablation_encoder_ts': ablation_encoder_ts,
        'ablation_encoder_se': ablation_encoder_se,
        'ablation_encoder_ts_fe': ablation_encoder_ts_fe,
        # 'ablation_encoder_ts_fe_nonull': ablation_encoder_ts_fe_nonull,
        # 'ablation_encoder_ts_fe_nonull_notime': ablation_encoder_ts_fe_nonull_notime,
        'ablation_encoder_stt_se': ablation_encoder_stt_se,
        # 'ablation_encoder_stt_se_nonull': ablation_encoder_stt_se_nonull,
        'ablation_encoder_se_se': ablation_encoder_se_se,
        # 'ablation_encoder_se_se_nonull': ablation_encoder_se_se_nonull,
        'ablation_encoder_stt_mts_e': ablation_encoder_stt_mts_e,
    }


def single_test_ablation(
        path_params: dict,
        prep_params: dict,
        eval_params: dict,
        model_params: dict,
        res_dir: str,
        data_dir: str,
        model_dir: str,
        ablation_test: str,
):

    subset = os.path.basename(path_params['ex_filename']).replace('subset_agg_', '').replace('.csv', '')
    nan_percentage = path_params['nan_percentage']
    num_fut = prep_params['ts_params']['num_fut']

    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    out_name = f"{path_params['type']}_{subset}_nan{int(nan_percentage * 10)}_nf{num_fut}"
    results_path = os.path.join(res_dir, f"{out_name}_{ablation_test}.csv")
    pickle_path = os.path.join(data_dir, f"{out_name}.pickle")
    checkpoint_path = os.path.join(model_dir, f"{out_name}_{os.getpid()}")

    print(f"{os.getpid()}_{out_name}: Preparing to run ablation test:", ablation_test)
    print(f"{os.getpid()}_{out_name}: Saving results in", results_path)
    print(f"{os.getpid()}_{out_name}: Saving pickle in", pickle_path)
    print(f"{os.getpid()}_{out_name}: Saving model in", checkpoint_path)

    results = dict()

    with open(pickle_path, "rb") as f:
        train_test_dict = pickle.load(f)

    # selected_model = train_test_dict['params']['model_params']['model_type'][:3].upper()

    results_name_mapping = {
        no_ablation: 'STT',
        ablation_embedder_no_time: 'STT w/o time enc',
        ablation_embedder_no_null: 'STT w/o null enc',
        ablation_embedder_no_time_null: 'STT w/o time null enc',
        ablation_encoder_t: 'T',
        ablation_encoder_s: 'S',
        ablation_encoder_e: 'E',
        ablation_encoder_te: 'TE',
        ablation_encoder_ts: 'TS',
        ablation_encoder_se: 'SE',
        ablation_encoder_ts_fe: 'TS_FE',
        # ablation_encoder_ts_fe_nonull: 'TS_FE nonull',
        # ablation_encoder_ts_fe_nonull_notime: 'TS_FE nonull notime',
        ablation_encoder_stt_se: 'STT_SE',
        # ablation_encoder_stt_se_nonull: 'STT_SE nonull',
        ablation_encoder_se_se: 'SE_SE',
        # ablation_encoder_se_se_nonull: 'SE_SE nonull',
        ablation_encoder_stt_mts_e: 'STT_MTS_E',
    }

    # Configure ablation test
    ablation_function = ablation_tests_mapping[ablation_test]
    train_test_dict = ablation_function(train_test_dict)

    # Exec ablation test
    print(f"{os.getpid()}_{out_name}: Launching {ablation_test} using model "
          f"{train_test_dict['params']['model_params']['model_type']}")
    results_name = results_name_mapping[ablation_function]

    """train_test_dict['x_train'] = train_test_dict['x_train'][:100, :, :]
    train_test_dict['spt_train'] = [data[:100, :, :] for data in train_test_dict['spt_train']]
    train_test_dict['exg_train'] = train_test_dict['exg_train'][:100, :, :]
    train_test_dict['y_train'] = train_test_dict['y_train'][:100, :]"""

    results[results_name] = model_step(train_test_dict, train_test_dict['params']['model_params'], checkpoint_path)
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(results_path, index=True)

    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)

    return results_df
