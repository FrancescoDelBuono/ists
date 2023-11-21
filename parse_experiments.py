import argparse
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser('FDB Parser')
# train configs
parser.add_argument('--ignore-incomplete', action='store_true', default=False,
                    help="Ignores incomplete model runs instead of raising an Exception.")

parser.add_argument('--models', nargs='+', default=['GRU-D', 'CRU', 'mTAN', 'ISTS'],
                    help="List of models to gather logs for.")

parser.add_argument('--folder', type=str, default='.',
                    help="Folder containing the logs to parse if you prefer to skip the default behaviour.")

parser.add_argument('--paper_table', action='store_true', default=False,
                    help="Prepares data in a format useful for reporting it in a paper.")

parser.add_argument('--metric', type=str, default='Test_MAE',
                    help="Metric to extract when using the --paper_table flag.")

parser.add_argument('--merge', action='store_true', default=False,
                    help="Merge the results of ISTS and the other models in a single dataframe.")

parser.add_argument('--fill_old', action='store_true', default=False,
                    help="Fill the table with old results first, then add the new ones.")

parser.add_argument('--ablation', action='store_true', default=False,
                    help="Parse the ablation experiments results.")

parser.add_argument('--debug_model', type=str, default=None, nargs='+',
                    help="Activate debug messages for a specific model.")

args = parser.parse_args()

folders = {
    'GRU-D': 'GRU-D',
    'CRU': 'Continuous-Recurrent-Units',
    'mTAN': 'mTAN',
    'ISTS': 'ists'
}

file_start = {
    'GRU-D': 'grud_output_',
    'CRU': 'cru_output_',
    'mTAN': 'mtan_output_',
    'ISTS': 'ists_output_'
}

models = ['t', 's', 'e', 'ts', 'te', 'se'] if args.ablation else ['ISTS', 'GRU-D', 'CRU', 'mTAN']

if args.ablation:
    df_dict = {model: pd.DataFrame() for model in models}
else:
    df_dict = {model: pd.DataFrame() for model in args.models}

loss_df = pd.DataFrame()


def parse_ists(file):
    global loss_df, df_dict

    if not args.ablation:
        df = df_dict['ISTS']
    else:
        df = None

    custom_nan = -999.0

    if not args.ablation:
        if df.empty:
            for column in [f'{mode}_{metric}' for mode in ['Train', 'Test'] for metric in ['R2', 'MSE', 'MAE']]:
                df[column] = custom_nan

    if loss_df.empty:
        for column in [f'{mode}_{metric}' for mode in ['Loss', 'Val_Loss'] for metric in range(20)]:
            loss_df[column] = custom_nan

    whitespace = ' '
    num_fut = nan_num = subset = model_type = None

    with open(file, 'r') as log_file:
        info_line = log_file.readline()

    info_line = info_line.split(whitespace)

    dataset_name = info_line[1]

    for info in info_line:
        if 'num_fut' in info:
            num_fut = info.split('=')[-1]
        elif 'nan_num' in info:
            nan_num = int(float(info.split('=')[-1]) * 10)
        elif 'subset' in info:
            subset = info.split('=')[-1].split('.csv')[0].split('subset_agg_')[-1]
        elif 'model_type' in info:
            model_type = info.split('=')[-1].strip()

    if args.ablation:
        df = df_dict[model_type]

        if df.empty:
            for column in [f'{mode}_{metric}' for mode in ['Train', 'Test'] for metric in ['R2', 'MSE', 'MAE']]:
                df[column] = custom_nan

    with open(file, 'r') as log_file:
        idx_string = f"{dataset_name}_{subset}_nan{nan_num}_nf{num_fut}_{model_type}"
        if not args.fill_old or idx_string not in df.index:
            df.loc[idx_string] = [custom_nan] * len(df.columns)
            loss_df.loc[idx_string] = [custom_nan] * len(loss_df.columns)

        for line in log_file:
            if 'test_r2' in line and 'train_r2' in line:
                metrics = eval(line)

                df.loc[idx_string, ['Train_R2', 'Test_R2']] = [metrics['train_r2'], metrics['test_r2']]
                df.loc[idx_string, ['Train_MSE', 'Test_MSE']] = [metrics['train_mse'], metrics['test_mse']]
                df.loc[idx_string, ['Train_MAE', 'Test_MAE']] = [metrics['train_mae'], metrics['test_mae']]

                for epoch, (loss, val_loss) in enumerate(zip(metrics['loss'], metrics['val_loss'])):
                    try:
                        loss = float(loss)
                        val_loss = float(val_loss)
                        loss_df.loc[idx_string, [f'Loss_{epoch}', f'Val_Loss_{epoch}']] = [loss, val_loss]

                    except ValueError:
                        loss_df.loc[idx_string, [f'Loss_{epoch}', f'Val_Loss_{epoch}']] = [np.nan, np.nan]

    if args.ablation:
        df_dict[model_type] = df
    else:
        # maybe superfluous
        df_dict['ISTS'] = df


def parse_model(model, file):
    custom_nan = -999.0
    df = df_dict[model]
    debug = args.debug_model and model in args.debug_model

    if df.empty:
        for column in [f'{mode}_{metric}' for mode in ['Train', 'Valid', 'Test'] for metric in ['R2', 'MSE', 'MAE']]:
            df[column] = np.nan

    dataset_name = file.split(file_start[model])[-1]
    dataset_name = dataset_name.split('.pickle.txt' if 'pickle' in file else '.txt')[0]

    # subset = underscore.join(dataset_name.split(underscore)[:-1])
    # num_fut = dataset_name.split(underscore)[-1].split('nf')[-1]

    with open(file, 'r') as log_file:
        metrics = log_file.readlines()[-20:]

        if not metrics:
            if not args.fill_old or dataset_name not in df.index:
                df.loc[dataset_name] = [np.nan] * len(df.columns)
            return  # file is not complete or empty

        if 'ValueError' in metrics[-1]:  # if error save NaN
            if dataset_name not in df.index:
                for mode in ['Train', 'Valid', 'Test']:
                    df.loc[dataset_name, [f'{mode}_R2', f'{mode}_MSE', f'{mode}_MAE']] = custom_nan

                if debug:
                    print("DEBUG: metrics[-1]:", metrics[-1])
                    print("DEBUG: df.loc[dataset_name]:", df.loc[dataset_name])

        else:  # parse the file with the correct modality per model
            if model == 'GRU-D':
                try:
                    if 'Performance metrics:' in metrics[-6] or 'Performance metrics:' in metrics[-5]:
                        if 'Performance metrics:' in metrics[-6]:
                            metrics = metrics[-5:-2]
                        else:
                            metrics = metrics[-4:-1]

                        for line, metric in zip(metrics, ['R2', 'MAE', 'MSE']):
                            if debug:
                                print("DEBUG: line:", line, "metric:", metric)
                            # train, valid, test
                            line = line.strip().split(f'{metric} score: ')[-1]
                            try:
                                metrics = eval(line)

                            except SyntaxError as se:
                                if not args.ignore_incomplete:
                                    raise SyntaxError(f'Could not eval {file}: {se}.')
                                continue

                            if debug:
                                print("DEBUG: metrics:", metrics)

                            if isinstance(metrics, list):
                                if debug:
                                    print("DEBUG: dataset_name in df.index:", dataset_name in df.index)
                                if dataset_name not in df.index:
                                    df.loc[dataset_name,
                                           [f'Train_{metric}', f'Valid_{metric}', f'Test_{metric}']] = metrics
                                else:
                                    if metric == 'R2':
                                        get_best = max
                                    else:
                                        get_best = min

                                    saved_train_metric = df.loc[dataset_name, f'Train_{metric}']
                                    if saved_train_metric != custom_nan and not np.isnan(saved_train_metric):
                                        df.loc[dataset_name,
                                               f'Train_{metric}'] = get_best(metrics[0],
                                                                             df.loc[dataset_name, f'Train_{metric}'])
                                    else:
                                        df.loc[dataset_name,
                                               f'Train_{metric}'] = metrics[0]

                                    saved_valid_metric = df.loc[dataset_name, f'Valid_{metric}']
                                    if saved_valid_metric != custom_nan and not np.isnan(saved_valid_metric):
                                        df.loc[dataset_name,
                                               f'Valid_{metric}'] = get_best(metrics[1],
                                                                             df.loc[dataset_name, f'Valid_{metric}'])
                                    else:
                                        df.loc[dataset_name,
                                               f'Valid_{metric}'] = metrics[1]

                                    saved_test_metric = df.loc[dataset_name, f'Test_{metric}']
                                    if saved_test_metric != custom_nan and not np.isnan(saved_test_metric):
                                        df.loc[dataset_name,
                                               f'Test_{metric}'] = get_best(metrics[2],
                                                                            df.loc[dataset_name, f'Test_{metric}'])
                                    else:
                                        df.loc[dataset_name,
                                               f'Test_{metric}'] = metrics[2]
                            else:
                                raise ValueError(f'Could not parse {file}: unknown metric format.')

                    else:
                        print(f"ValueError: Could not parse {file}: unknown structure.")
                        if debug:
                            print("DEBUG: metrics", metrics)
                        if not args.ignore_incomplete:
                            raise ValueError(f'Could not parse {file}: unknown structure.')

                except IndexError:
                    print(f'IndexError: Could not parse {file}: unknown structure.')
                    if dataset_name not in df.index:
                        df.loc[dataset_name] = [np.nan] * len(df.columns)

                if debug:
                    if dataset_name in df.index:
                        print("DEBUG: df.loc[dataset_name]:", df.loc[dataset_name])
                    else:
                        print(f"DEBUG: {dataset_name} not in index.")

            elif model == 'CRU':
                try:
                    metrics = metrics[-3:]
                    if 'Train MAE' in metrics[-1]:

                        for line, metric in zip(metrics, ['R2', 'MSE', 'MAE']):
                            train_valid, test = line.split(f'Test RMSE:' if 'RMSE' in line
                                                           else f'Test {metric}:')

                            train_valid = train_valid.split(', ')[0].split(f'Train RMSE: ' if 'RMSE' in line
                                                                           else f'Train {metric}: ')[-1]
                            train_valid = float(train_valid)
                            test = float(test.strip())

                            if dataset_name not in df.index:
                                df.loc[dataset_name,
                                       [f'Train_{metric}', f'Valid_{metric}']] = train_valid
                                df.loc[dataset_name, f'Test_{metric}'] = test
                            else:
                                if metric == 'R2':
                                    get_best = max
                                else:
                                    get_best = min

                                saved_train_metric = df.loc[dataset_name, f'Train_{metric}']
                                if saved_train_metric != custom_nan and not np.isnan(saved_train_metric):
                                    df.loc[dataset_name,
                                           f'Train_{metric}'] = get_best(train_valid,
                                                                         df.loc[dataset_name, f'Train_{metric}'])
                                else:
                                    df.loc[dataset_name,
                                           f'Train_{metric}'] = train_valid

                                saved_valid_metric = df.loc[dataset_name, f'Valid_{metric}']
                                if saved_valid_metric != custom_nan and not np.isnan(saved_valid_metric):
                                    df.loc[dataset_name,
                                           f'Valid_{metric}'] = get_best(train_valid,
                                                                         df.loc[dataset_name, f'Valid_{metric}'])
                                else:
                                    df.loc[dataset_name,
                                           f'Valid_{metric}'] = train_valid

                                saved_test_metric = df.loc[dataset_name, f'Test_{metric}']
                                if saved_test_metric != custom_nan and not np.isnan(saved_test_metric):
                                    df.loc[dataset_name,
                                           f'Test_{metric}'] = get_best(test,
                                                                        df.loc[dataset_name, f'Test_{metric}'])
                                else:
                                    df.loc[dataset_name,
                                           f'Test_{metric}'] = test

                    else:
                        print(f"ValueError: Could not parse {file}: unknown structure.")
                        if debug:
                            print("DEBUG: metrics", metrics)
                        if not args.ignore_incomplete:
                            raise ValueError(f'Could not parse {file}: unknown structure.')
                except IndexError:
                    print(f'IndexError: Could not parse {file}: unknown structure.')
                    if dataset_name not in df.index:
                        df.loc[dataset_name] = [np.nan] * len(df.columns)

                if debug:
                    if dataset_name in df.index:
                        print("DEBUG: df.loc[dataset_name]:", df.loc[dataset_name])
                    else:
                        print(f"DEBUG: {dataset_name} not in index.")

            elif model == 'mTAN':
                try:
                    metrics = metrics[-3].split(', ')

                    if debug:
                        print("DEBUG: metrics:", metrics)

                    if len(metrics) > 5 and 'train_mse' in metrics[4]:
                        train_mse = metrics[4].split('train_mse: ')[-1]
                        train_mae = metrics[5].split('train_mae: ')[-1]
                        train_r2 = metrics[6].split('train_r2: ')[-1]

                        train_mse = float(train_mse) if train_mse != 'nan' else custom_nan
                        train_mae = float(train_mae) if train_mae != 'nan' else custom_nan
                        train_r2 = float(train_r2) if train_r2 != 'nan' else custom_nan

                        if dataset_name not in df.index:
                            df.loc[dataset_name, ['Train_MSE', 'Train_MAE', 'Train_R2']] = \
                                [train_mse, train_mae, train_r2]

                            if debug:
                                print("DEBUG: assigned train values directly.")
                        else:
                            saved_train_mse = df.loc[dataset_name, 'Train_MSE']
                            if saved_train_mse != custom_nan and not np.isnan(saved_train_mse):
                                if debug:
                                    print("Comparing train mse:", df.loc[dataset_name, 'Train_MSE'], "and", train_mse)
                                df.loc[dataset_name, 'Train_MSE'] = min(df.loc[dataset_name, 'Train_MSE'], train_mse)
                            else:
                                df.loc[dataset_name, 'Train_MSE'] = train_mse

                            saved_train_mae = df.loc[dataset_name, 'Train_MAE']
                            if saved_train_mae != custom_nan and not np.isnan(saved_train_mae):
                                if debug:
                                    print("Comparing train mae:", df.loc[dataset_name, 'Train_MAE'], "and", train_mae)
                                df.loc[dataset_name, 'Train_MAE'] = min(df.loc[dataset_name, 'Train_MAE'], train_mae)
                            else:
                                df.loc[dataset_name, 'Train_MAE'] = train_mae

                            saved_train_r2 = df.loc[dataset_name, 'Train_R2']
                            if saved_train_r2 != custom_nan and not np.isnan(saved_train_r2):
                                if debug:
                                    print("Comparing train r2:", df.loc[dataset_name, 'Train_R2'], "and", train_r2)
                                df.loc[dataset_name, 'Train_R2'] = max(df.loc[dataset_name, 'Train_R2'], train_r2)
                            else:
                                df.loc[dataset_name, 'Train_R2'] = train_r2

                        val_mse = metrics[9].split('val_mse: ')[-1]
                        val_mae = metrics[10].split('val_mae: ')[-1]
                        val_r2 = metrics[11].split('val_r2: ')[-1]

                        val_mse = float(val_mse) if val_mse != 'nan' else custom_nan
                        val_mae = float(val_mae) if val_mae != 'nan' else custom_nan
                        val_r2 = float(val_r2) if val_r2 != 'nan' else custom_nan

                        if debug:
                            print("DEBUG: metrics[9]:", metrics[9], "to float: ", val_mse)
                            print("DEBUG: metrics[10]:", metrics[10], "to float: ", val_mae)
                            print("DEBUG: metrics[11]:", metrics[11], "to float: ", val_r2)

                        if dataset_name not in df.index:
                            df.loc[dataset_name, ['Valid_MSE', 'Valid_MAE', 'Valid_R2']] = \
                                [val_mse, val_mae, val_r2]

                            if debug:
                                print("DEBUG: assigned valid values directly.")
                        else:
                            saved_val_mse = df.loc[dataset_name, 'Valid_MSE']
                            if saved_val_mse != custom_nan and not np.isnan(saved_val_mse):
                                if debug:
                                    print("Comparing valid mse:", df.loc[dataset_name, 'Valid_MSE'], "and", val_mse)
                                df.loc[dataset_name, 'Valid_MSE'] = min(df.loc[dataset_name, 'Valid_MSE'], val_mse)
                            else:
                                df.loc[dataset_name, 'Valid_MSE'] = val_mse

                            saved_val_mae = df.loc[dataset_name, 'Valid_MAE']
                            if saved_val_mae != custom_nan and not np.isnan(saved_val_mae):
                                if debug:
                                    print("Comparing valid mae:", df.loc[dataset_name, 'Valid_MAE'], "and", val_mae)
                                df.loc[dataset_name, 'Valid_MAE'] = min(df.loc[dataset_name, 'Valid_MAE'], val_mae)
                            else:
                                df.loc[dataset_name, 'Valid_MAE'] = val_mae

                            saved_val_r2 = df.loc[dataset_name, 'Valid_R2']
                            if saved_val_r2 != custom_nan and not np.isnan(saved_val_r2):
                                if debug:
                                    print("Comparing valid r2:", df.loc[dataset_name, 'Valid_R2'], "and", val_r2)
                                df.loc[dataset_name, 'Valid_R2'] = max(df.loc[dataset_name, 'Valid_R2'], val_r2)
                            else:
                                df.loc[dataset_name, 'Valid_R2'] = val_r2

                        test_mse = metrics[13].split('test_mse: ')[-1]
                        test_mae = metrics[14].split('test_mae: ')[-1]
                        test_r2 = metrics[15].split('test_r2: ')[-1].strip()

                        test_mse = float(test_mse) if test_mse != 'nan' else custom_nan
                        test_mae = float(test_mae) if test_mae != 'nan' else custom_nan
                        test_r2 = float(test_r2) if test_r2 != 'nan' else custom_nan

                        if debug:
                            print("DEBUG: metrics[13]:", metrics[13], "to float: ", test_mse)
                            print("DEBUG: metrics[14]:", metrics[14], "to float: ", test_mae)
                            print("DEBUG: metrics[15]:", metrics[15], "to float: ", test_r2)

                        if dataset_name not in df.index:
                            df.loc[dataset_name, ['Test_MSE', 'Test_MAE', 'Test_R2']] = \
                                [test_mse, test_mae, test_r2]

                            if debug:
                                print("DEBUG: assigned test values directly.")
                        else:
                            saved_test_mse = df.loc[dataset_name, 'Test_MSE']
                            if saved_test_mse != custom_nan and not np.isnan(saved_test_mse):
                                if debug:
                                    print("Comparing test mse:", df.loc[dataset_name, 'Test_MSE'], "and", test_mse)
                                df.loc[dataset_name, 'Test_MSE'] = min(df.loc[dataset_name, 'Test_MSE'], test_mse)
                            else:
                                df.loc[dataset_name, 'Test_MSE'] = test_mse

                            saved_test_mae = df.loc[dataset_name, 'Test_MAE']
                            if saved_test_mae != custom_nan and not np.isnan(saved_test_mae):
                                if debug:
                                    print("Comparing test mae:", df.loc[dataset_name, 'Test_MAE'], "and", test_mae)
                                df.loc[dataset_name, 'Test_MAE'] = min(df.loc[dataset_name, 'Test_MAE'], test_mae)
                            else:
                                df.loc[dataset_name, 'Test_MAE'] = test_mae

                            saved_test_r2 = df.loc[dataset_name, 'Test_R2']
                            if saved_test_r2 != custom_nan and not np.isnan(saved_test_r2):
                                if debug:
                                    print("Comparing test r2:", df.loc[dataset_name, 'Test_R2'], "and", test_r2)
                                df.loc[dataset_name, 'Test_R2'] = max(df.loc[dataset_name, 'Test_R2'], test_r2)
                            else:
                                df.loc[dataset_name, 'Test_R2'] = test_r2

                    else:
                        print(f"ValueError: Could not parse {file}: unknown structure.")
                        if debug:
                            print("DEBUG: metrics", metrics)
                        if not args.ignore_incomplete:
                            raise ValueError(f'Could not parse {file}: unknown structure.')

                except IndexError:
                    print(f'IndexError: Could not parse {file}: unknown structure.')
                    df.loc[dataset_name] = [np.nan] * len(df.columns)

                if debug:
                    if dataset_name in df.index:
                        print("DEBUG: df.loc[dataset_name]:", df.loc[dataset_name])
                    else:
                        print(f"DEBUG: {dataset_name} not in index.")

            else:
                raise ValueError(f'Unknown model {model}.')

    # maybe superfluous
    df_dict[model] = df


def main():
    global models

    if args.fill_old:
        parse_logs('./backup_logs')

    parse_logs(args.folder)

    if args.paper_table:
        nan_nums = [0.0, 0.2, 0.5, 0.8]
        num_futs = [7, 14, 30, 60]

        columns = pd.MultiIndex.from_tuples([(model, nan_num) for nan_num in nan_nums for model in models],
                                            names=['model', 'nan_num'])

        complete_dfs = {num_fut: pd.DataFrame(columns=columns) for num_fut in num_futs}
        for num_fut, complete_df in complete_dfs.items():
            complete_df.rename_axis(index='Dataset', inplace=True)
            # example: french_subset_agg_th1_1_0.2_nf14_sttransformer
            for model, df in df_dict.items():
                try:
                    subset_df = df.loc[df.index.str.contains(f'nf{num_fut}')]
                except AttributeError as ae:
                    print(model, df, sep='\n')
                    raise ae

                parameters = subset_df.index.to_series().apply(lambda dataset_: dataset_.split('_'))
                subset_idx = -1
                nan_idx = -1

                for dataset_params in parameters:
                    original_dataset_name = '_'.join(dataset_params)

                    for param in dataset_params:
                        if param.startswith('th'):
                            subset_idx = dataset_params.index(param)
                        if param.startswith('nan'):
                            nan_idx = dataset_params.index(param)

                    dataset = dataset_params[subset_idx - 1]
                    subset = '_'.join(dataset_params[subset_idx:nan_idx])
                    nan_num = float(dataset_params[nan_idx].split('nan')[-1]) / 10

                    complete_df.loc[f"{dataset}_{subset}",
                                    (model, nan_num)] = subset_df.loc[original_dataset_name, args.metric]

            complete_df.sort_index(inplace=True)
            # complete_df.sort_index(axis=1, level=['model', 'nan_num'], ascending=[True, True], inplace=True)
            # force order of columns
            if args.ablation:
                complete_df = complete_df.loc[:, [('t', 0.0), ('t', 0.2), ('t', 0.5), ('t', 0.8),
                                                  ('s', 0.0), ('s', 0.2), ('s', 0.5), ('s', 0.8),
                                                  ('e', 0.0), ('e', 0.2), ('e', 0.5), ('e', 0.8),
                                                  ('ts', 0.0), ('ts', 0.2), ('ts', 0.5), ('ts', 0.8),
                                                  ('te', 0.0), ('te', 0.2), ('te', 0.5), ('te', 0.8),
                                                  ('se', 0.0), ('se', 0.2), ('se', 0.5), ('se', 0.8)]]
            else:
                complete_df = complete_df.loc[:, [('ISTS', 0.0), ('ISTS', 0.2), ('ISTS', 0.5), ('ISTS', 0.8),
                                                  ('GRU-D', 0.0), ('GRU-D', 0.2), ('GRU-D', 0.5), ('GRU-D', 0.8),
                                                  ('mTAN', 0.0), ('mTAN', 0.2), ('mTAN', 0.5), ('mTAN', 0.8),
                                                  ('CRU', 0.0), ('CRU', 0.2), ('CRU', 0.5), ('CRU', 0.8)]]

            complete_filename = (f'{"ablation" if args.ablation else "complete"}'
                                 f'_results_{"old_filled_" if args.fill_old else ""}'
                                 f'nf{num_fut}_{args.metric}.csv')
            complete_df.to_csv(complete_filename)

    else:
        for model, df in df_dict.items():
            df.rename_axis(index='Dataset', inplace=True)
            df.to_csv(f'{model}_{"ablation_" if args.ablation else ""}results.csv')

    if not loss_df.empty:
        loss_df.rename_axis(index='Dataset', inplace=True)
        loss_df.to_csv(f'ists_{"ablation_" if args.ablation else ""}losses.csv')


def parse_logs(folder: str):
    wdir = os.getcwd()
    for model in args.models:
        if folder != '.':
            print(f"Moving to {folder}.")
            os.chdir(folder)

        # if I'm not in a model folder already
        if wdir.split('/')[-1] != folders[model] and os.path.exists(f'./{folders[model]}'):
            print(f"Parsing files in {folders[model]}.")
            os.chdir(folders[model])
        else:
            print(f"Parsing files in {wdir}.")

        for file in os.listdir():
            if file.endswith('.err'):
                continue

            if file.startswith(file_start[model]):
                print(f'Parsing {file}...')

                if model == 'ISTS' or args.ablation:
                    parse_ists(file)

                else:
                    parse_model(model, file)

        os.chdir(wdir)


def merge():
    ists_results_path = './ists_complete_results'
    old_filled_files = [file for file in os.listdir() if 'old_filled' in file and file.endswith('.csv')]
    for csv_file in [file for file in os.listdir(ists_results_path) if file.endswith('.csv')]:
        ists_results = pd.read_csv(os.path.join(ists_results_path, csv_file), index_col=0, header=[0, 1])

        if args.ablation:
            ablation_results = pd.read_csv(csv_file.replace('complete', 'ablation'),
                                           index_col=0, header=[0, 1])

            for idx, nan_num in enumerate(('0.0', '0.2', '0.5', '0.8')):
                ablation_results.insert(idx, ('sttransformer', nan_num), ists_results.loc[:, ('ISTS', nan_num)])

            ablation_results.to_csv(csv_file.replace('complete', 'ablation'))

        else:
            others_results = pd.read_csv(csv_file, index_col=0, header=[0, 1])

            for nan_num in ('0.0', '0.2', '0.5', '0.8'):
                others_results.loc[:, ('ISTS', nan_num)] = ists_results.loc[:, ('ISTS', nan_num)]

            others_results.to_csv(csv_file)

            if old_filled_files:
                for file in old_filled_files:
                    if ''.join(file.split('_old_filled')) == csv_file:
                        print(f"Merging with old_filled results {file}.")
                        old_filled_results = pd.read_csv(file, index_col=0, header=[0, 1])

                        for nan_num in ('0.0', '0.2', '0.5', '0.8'):
                            old_filled_results.loc[:, ('ISTS', nan_num)] = ists_results.loc[:, ('ISTS', nan_num)]

                        old_filled_results.to_csv(file)

                        break


if __name__ == '__main__':
    if args.merge:
        merge()
    else:
        main()
