import os
import pickle

import pandas as pd

from pipeline import data_step, model_step, parse_params, change_params


def no_ablation(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "sttransformer"
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

    ablations_mapping = {
        'STT': no_ablation,
    }

    if ablation_embedder:
        ablations_mapping.update({
            'STT w/o time enc': ablation_embedder_no_time,
            'STT w/o null enc': ablation_embedder_no_null,
            'STT w/o time null enc': ablation_embedder_no_time_null,
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

    for name, func in ablations_mapping.items():
        print(f'\n{name}')
        with open(pickle_path, "rb") as f:
            train_test_dict = pickle.load(f)
        train_test_dict = func(train_test_dict)
        results[name] = model_step(train_test_dict, train_test_dict['params']['model_params'], checkpoint_path)
        pd.DataFrame(results).T.to_csv(results_path, index=True)

    return pd.DataFrame(results).T


def main():
    res_dir = './output/results'
    data_dir = './output/pickle'
    model_dir = './output/model'

    path_params, prep_params, eval_params, model_params = parse_params()
    # path_params = change_params(path_params, '../../data', '../../Dataset/AdbPo')

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

    print('Hello World!')


if __name__ == '__main__':
    main()
