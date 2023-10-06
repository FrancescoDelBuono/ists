import copy
from datetime import datetime

import pandas as pd
from sklearn.model_selection import ParameterGrid

from parallel_pipeline import data_step, model_step, get_params, parse_params


def my_model_search(path_params, prep_params, eval_params, model_params):
    grid_search_params = {
        'nn_params': [
            {
                'kernel_size': 3,
                'd_model': 32,
                'num_heads': 8,
                'dff': 64,
                'fff': 32,
                'exg_cnn': True,
                'spt_cnn': True,
                'time_cnn': True,
                'dropout_rate': 0.2,
                'num_layers': 1,
                'with_cross': True,
            },
            {
                'kernel_size': 3,
                'd_model': 128,
                'num_heads': 8,
                'dff': 256,
                'fff': 128,
                'exg_cnn': True,
                'spt_cnn': True,
                'time_cnn': True,
                'dropout_rate': 0.2,
                'num_layers': 4,
                'with_cross': True,
            },
            {
                'kernel_size': 3,
                'd_model': 32,
                'num_heads': 8,
                'dff': 64,
                'fff': 32,
                'dropout_rate': 0.2,
                'num_layers': 1,
                'with_cross': False,
            },
            {
                'kernel_size': 3,
                'd_model': 128,
                'num_heads': 8,
                'dff': 256,
                'fff': 128,
                'exg_cnn': True,
                'spt_cnn': True,
                'time_cnn': True,
                'dropout_rate': 0.2,
                'num_layers': 4,
                'with_cross': False,
            },
        ],
        "transform_type": ["standard"],  # , "minmax"],
        "epochs": [50],  # , 100],
        "loss": ['mse'],  # , "mae"],
        "exg_num_past": [60, 72, 104],
        "spt_num_past": [36, 48],
        "spt_num_spt": [2, 5, 7],
        "x_num_past": [36],
    }

    configs = []
    params = list(ParameterGrid(grid_search_params))
    for param in ParameterGrid(grid_search_params):

        # Prep params
        prep_params['ts_params']['num_past'] = param['x_num_past']
        prep_params['spt_params']['num_past'] = param['spt_num_past']
        prep_params['spt_params']['num_spt'] = param['spt_num_spt']
        prep_params['exg_params']['num_past'] = param['exg_num_past']

        # Model params
        model_params['transform_type'] = param['transform_type']
        model_params['epochs'] = param['epochs']
        model_params['loss'] = param['loss']
        model_params['nn_params'] = param['nn_params']

        path_params_copy = copy.deepcopy(path_params)
        prep_params_copy = copy.deepcopy(prep_params)
        eval_params_copy = copy.deepcopy(eval_params)
        model_params_copy = copy.deepcopy(model_params)

        config = (
            path_params_copy,
            prep_params_copy,
            eval_params_copy,
            model_params_copy
        )
        configs.append(config)
    return configs, params


def main():
    # Read input base params
    # path_params, prep_params, eval_params, model_params = get_params()
    path_params, prep_params, eval_params, model_params = parse_params()
    # Extract all possibilities
    configs, params = my_model_search(path_params, prep_params, eval_params, model_params)

    results_filename = './results.csv'

    results = []
    for i in range(len(configs)):
        param = params[i]  # Extract pipeline param

        print('\n\n' + '\u2500' * 80)
        print(f"{i}:\t {param}\n{datetime.now()}\n")

        # Extract pipeline configuration params
        path_params, prep_params, eval_params, model_params = configs[i]
        # Data step
        train_test_dict = data_step(path_params, prep_params, eval_params)
        # Model step
        res = model_step(train_test_dict, model_params)

        # Update result dictionary with pipeline params
        res.update(param)
        # Save step results
        results.append(res)

        # Save dataframe
        df_res = pd.DataFrame(results)
        df_res.to_csv(results_filename, index=False)

        print(f'\n{param}\n')

    print('Hello World!')


if __name__ == '__main__':
    main()
