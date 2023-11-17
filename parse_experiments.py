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

parser.add_argument('--folder', type=str, default=None,
                    help="Folder containing the logs to parse if you prefer to skip the default behaviour.")

parser.add_argument('--paper_table', action='store_true', default=False,
                    help="Prepares data in a format useful for reporting it in a paper.")

parser.add_argument('--metric', type=str, default='Test_MAE',
                    help="Metric to extract when using the --paper_table flag.")

parser.add_argument('--merge', action='store_true', default=False,
                    help="Merge the results of ISTS and the other models in a single dataframe.")

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

models = ['ISTS', 'GRU-D', 'CRU', 'mTAN']

df_dict = {model: pd.DataFrame() for model in args.models}
loss_df = pd.DataFrame()


def parse_ists(file):
    global loss_df, df_dict
    df = df_dict['ISTS']

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
            model_type = info.split('=')[-1]

    with open(file, 'r') as log_file:
        for line in log_file:
            if 'test_r2' in line and 'train_r2' in line:
                metrics = eval(line)

                df.loc[f"{dataset_name}_{subset}_nan{nan_num}_nf{num_fut}_{model_type}", ['Train_R2', 'Test_R2']] = \
                    [metrics['train_r2'], metrics['test_r2']]
                df.loc[f"{dataset_name}_{subset}_nan{nan_num}_nf{num_fut}_{model_type}", ['Train_MSE', 'Test_MSE']] = \
                    [metrics['train_mse'], metrics['test_mse']]
                df.loc[f"{dataset_name}_{subset}_nan{nan_num}_nf{num_fut}_{model_type}", ['Train_MAE', 'Test_MAE']] = \
                    [metrics['train_mae'], metrics['test_mae']]

                for epoch, (loss, val_loss) in enumerate(zip(metrics['loss'], metrics['val_loss'])):
                    loss_df.loc[f"{dataset_name}_{subset}_nan{nan_num}_nf{num_fut}_{model_type}",
                                [f'Loss_{epoch}', f'Val_Loss_{epoch}']] = [loss, val_loss]

    # maybe superfluous
    df_dict['ISTS'] = df


def parse_model(model, file):
    df = df_dict[model]

    dataset_name = file.split(file_start[model])[-1]
    dataset_name = dataset_name.split('.pickle.txt' if 'pickle' in file else '.txt')[0]

    # subset = underscore.join(dataset_name.split(underscore)[:-1])
    # num_fut = dataset_name.split(underscore)[-1].split('nf')[-1]

    with open(file, 'r') as log_file:
        metrics = log_file.readlines()[-20:]

        if not metrics:
            return  # file is not complete or empty

        if 'ValueError' in metrics[-1]:  # if error save NaN
            for mode in ['Train', 'Valid', 'Test']:
                df.loc[dataset_name, [f'{mode}_R2', f'{mode}_MSE', f'{mode}_MAE']] = np.nan

        else:  # parse the file with the correct modality per model
            if model == 'GRU-D':
                if 'Performance metrics:' in metrics[-6]:
                    metrics = metrics[-5:-2]

                    for line, metric in zip(metrics, ['R2', 'MAE', 'MSE']):
                        # train, valid, test
                        line = line.strip().split(f'{metric} score: ')[-1]
                        try:
                            metrics = eval(line)

                        except SyntaxError as se:
                            if not args.ignore_incomplete:
                                raise SyntaxError(f'Could not eval {file}: {se}.')
                            continue

                        if isinstance(metrics, list):
                            df.loc[dataset_name,
                                   [f'Train_{metric}', f'Valid_{metric}', f'Test_{metric}']] = metrics

                        else:
                            raise ValueError(f'Could not parse {file}: unknown metric format.')

                else:
                    if not args.ignore_incomplete:
                        raise ValueError(f'Could not parse {file}: unknown structure.')

            elif model == 'CRU':
                metrics = metrics[-3:]
                if 'Train MAE' in metrics[-1]:

                    for line, metric in zip(metrics, ['R2', 'MSE', 'MAE']):
                        train_valid, test = line.split(f'Test RMSE:' if 'RMSE' in line
                                                       else f'Test {metric}:')

                        train_valid = train_valid.split(', ')[0].split(f'Train RMSE: ' if 'RMSE' in line
                                                                       else f'Train {metric}: ')[-1]
                        train_valid = float(train_valid)
                        test = float(test.strip())

                        df.loc[dataset_name,
                               [f'Train_{metric}', f'Valid_{metric}']] = train_valid
                        df.loc[dataset_name, f'Test_{metric}'] = test
                else:
                    if not args.ignore_incomplete:
                        raise ValueError(f'Could not parse {file}: unknown structure.')

            elif model == 'mTAN':
                metrics = metrics[-3].split(', ')

                if len(metrics) > 5 and 'train_mse' in metrics[4]:
                    train_mse = metrics[4].split('train_mse: ')[-1]
                    train_mae = metrics[5].split('train_mae: ')[-1]
                    train_r2 = metrics[6].split('train_r2: ')[-1]

                    train_mse = float(train_mse) if train_mse != 'nan' else np.nan
                    train_mae = float(train_mae) if train_mae != 'nan' else np.nan
                    train_r2 = float(train_r2) if train_r2 != 'nan' else np.nan

                    df.loc[dataset_name, ['Train_MSE', 'Train_MAE', 'Train_R2']] = \
                        [train_mse, train_mae, train_r2]
                    val_mse = metrics[9].split('val_mse: ')[-1]
                    val_mae = metrics[10].split('val_mae: ')[-1]
                    val_r2 = metrics[11].split('val_r2: ')[-1]

                    val_mse = float(val_mse) if val_mse != 'nan' else np.nan
                    val_mae = float(val_mae) if val_mae != 'nan' else np.nan
                    val_r2 = float(val_r2) if val_r2 != 'nan' else np.nan

                    df.loc[dataset_name, ['Valid_MSE', 'Valid_MAE', 'Valid_R2']] = \
                        [val_mse, val_mae, val_r2]
                    test_mse = metrics[13].split('test_mse: ')[-1]
                    test_mae = metrics[14].split('test_mae: ')[-1]
                    test_r2 = metrics[15].split('test_r2: ')[-1].strip()

                    test_mse = float(test_mse) if test_mse != 'nan' else np.nan
                    test_mae = float(test_mae) if test_mae != 'nan' else np.nan
                    test_r2 = float(test_r2) if test_r2 != 'nan' else np.nan

                    df.loc[dataset_name, ['Test_MSE', 'Test_MAE', 'Test_R2']] = \
                        [test_mse, test_mae, test_r2]

                else:
                    if not args.ignore_incomplete:
                        raise ValueError(f'Could not parse {file}: unknown structure.')

            else:
                raise ValueError(f'Unknown model {model}.')

    # maybe superfluous
    df_dict[model] = df


def main():
    # models = ['ISTS', 'GRU-D', 'mTAN', 'CRU']

    wdir = os.getcwd()

    for model in args.models:
        if args.folder:
            if args.folder != '.':
                os.chdir(args.folder)
        else:
            if wdir.split('/')[-1] != folders[model]:
                os.chdir(folders[model])

        for file in os.listdir():
            if file.endswith('.err'):
                continue

            if file.startswith(file_start[model]):
                print(f'Parsing {file}...')

                if model == 'ISTS':
                    parse_ists(file)

                else:
                    parse_model(model, file)

        os.chdir(wdir)

    if args.paper_table:
        columns = pd.MultiIndex.from_tuples([(model, nan_num) for nan_num in [0.0, 0.2, 0.5, 0.8] for model in models],
                                            names=['model', 'nan_num'])
        complete_dfs = {num_fut: pd.DataFrame(columns=columns) for num_fut in [7, 14, 30, 60]}
        for num_fut, complete_df in complete_dfs.items():
            complete_df.rename_axis(index='Dataset', inplace=True)
            # french_subset_agg_th1_1_0.2_nf14_sttransformer
            for model, df in df_dict.items():
                subset_df = df.loc[df.index.str.contains(f'nf{num_fut}')]
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
                    # num_fut = dataset_params[nan_idx + 1]
                    # model_type = dataset_params[-1]

                    complete_df.loc[f"{dataset}_{subset}",
                                    (model, nan_num)] = subset_df.loc[original_dataset_name, args.metric]
            complete_df: pd.DataFrame
            complete_df.sort_index(inplace=True)
            # complete_df.sort_index(axis=1, level=['model', 'nan_num'], ascending=[True, True], inplace=True)
            # force order of columns
            complete_df = complete_df.loc[:, [('ISTS', 0.0), ('ISTS', 0.2), ('ISTS', 0.5), ('ISTS', 0.8),
                                              ('GRU-D', 0.0), ('GRU-D', 0.2), ('GRU-D', 0.5), ('GRU-D', 0.8),
                                              ('mTAN', 0.0), ('mTAN', 0.2), ('mTAN', 0.5), ('mTAN', 0.8),
                                              ('CRU', 0.0), ('CRU', 0.2), ('CRU', 0.5), ('CRU', 0.8)]]
            complete_df.to_csv(f'complete_results_nf{num_fut}_{args.metric}.csv')

    else:
        for model, df in df_dict.items():
            df.rename_axis(index='Dataset', inplace=True)
            df.to_csv(f'{model}_results.csv')

    if not loss_df.empty:
        loss_df.rename_axis(index='Dataset', inplace=True)
        loss_df.to_csv('ists_losses.csv')


def merge():
    ists_results_path = './ists_complete_results'
    for csv_file in [file for file in os.listdir(ists_results_path) if file.endswith('.csv')]:
        ists_results = pd.read_csv(os.path.join(ists_results_path, csv_file), index_col=0, header=[0, 1])
        others_results = pd.read_csv(csv_file, index_col=0, header=[0, 1])

        others_results.loc[:, ('ISTS', 0.0)] = ists_results.loc[:, ('ISTS', 0.0)]
        others_results.loc[:, ('ISTS', 0.2)] = ists_results.loc[:, ('ISTS', 0.2)]
        others_results.loc[:, ('ISTS', 0.5)] = ists_results.loc[:, ('ISTS', 0.5)]
        others_results.loc[:, ('ISTS', 0.8)] = ists_results.loc[:, ('ISTS', 0.8)]

        others_results.to_csv(csv_file)


if __name__ == '__main__':
    if args.merge:
        merge()
    else:
        main()
