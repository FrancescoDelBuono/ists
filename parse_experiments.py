import argparse
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser('FDB Parser')
# train configs
parser.add_argument('--ignore-incomplete', action='store_true', default=False,
                    help="List of models to train and test separated by spaces.")

parser.add_argument('--models', nargs='+', default=['GRU-D', 'CRU', 'mTAN', 'ISTS'],
                    help="List of model logs to parse separated by spaces.")

parser.add_argument('--folder', type=str, default=None,
                    help="Folder containing the logs to parse if you prefer to skip the default behaviour.")

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

df_dict = {model: pd.DataFrame() for model in args.models}
loss_df = pd.DataFrame()


def parse_ists(file):
    global loss_df, df_dict
    df = df_dict['ISTS']

    whitespace = ' '
    num_fut = subset = model_type = None

    with open(file, 'r') as log_file:
        info_line = log_file.readline()

    info_line = info_line.split(whitespace)

    dataset_name = info_line[1]

    for info in info_line:
        if 'num_fut' in info:
            num_fut = info.split('=')[-1]
        elif 'subset' in info:
            subset = info.split('=')[-1].split('.csv')[0]
        elif 'model_type' in info:
            model_type = info.split('=')[-1]

    with (open(file, 'r') as log_file):
        for line in log_file:
            if 'test_r2' in line and 'train_r2' in line:
                metrics = eval(line)

                df.loc[f"{dataset_name}_{subset}_nf{num_fut}_{model_type}", ['Train_R2', 'Test_R2']] = \
                    [metrics['train_r2'], metrics['test_r2']]
                df.loc[f"{dataset_name}_{subset}_nf{num_fut}_{model_type}", ['Train_MSE', 'Test_MSE']] = \
                    [metrics['train_mse'], metrics['test_mse']]
                df.loc[f"{dataset_name}_{subset}_nf{num_fut}_{model_type}", ['Train_MAE', 'Test_MAE']] = \
                    [metrics['train_mae'], metrics['test_mae']]

                for epoch, (loss, val_loss) in enumerate(zip(metrics['loss'], metrics['val_loss'])):
                    loss_df.loc[f"{dataset_name}_{subset}_nf{num_fut}_{model_type}",
                                [f'Loss_{epoch}', f'Val_Loss_{epoch}']] = [loss, val_loss]

    # maybe superfluous
    df_dict['ISTS'] = df


def parse_model(model, file):
    df = df_dict[model]
    dataset_name = file.split(file_start[model])[-1]
    dataset_name = dataset_name.split('.pickle.txt' if 'pickle' in file else '.txt')[0]

    underscore = '_'

    subset = underscore.join(dataset_name.split(underscore)[:-1])
    num_fut = dataset_name.split(underscore)[-1].split('nf')[-1]

    if model != 'ISTS':
        with open(file, 'r') as log_file:
            metrics = log_file.readlines()[-20:]

            if 'ValueError' in metrics[-1]:  # if error save NaN
                for mode in ['Train', 'Valid', 'Test']:
                    df.loc[f"{dataset_name}_{subset}_{num_fut}", [f'{mode}_R2', f'{mode}_MSE', f'{mode}_MAE']] = np.nan

            else:  # parse the file with the correct modality per model
                if model == 'GRU-D':
                    if 'Performance metrics:' in metrics[-6]:
                        metrics = metrics[-5:-2]

                        for line, metric in zip(metrics, ['R2', 'MAE', 'MSE']):
                            # train, valid, test
                            line = line.strip().lstrip(f'{metric} score: ')
                            try:
                                metrics = eval(line)

                            except SyntaxError as se:
                                if not args.ignore_incomplete:
                                    raise SyntaxError(f'Could not eval {file}: {se}.')
                                continue

                            if isinstance(metrics, list):
                                df.loc[f"{dataset_name}_{subset}_{num_fut}",
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

                            train_valid = train_valid.rstrip(', ').lstrip(f'Train RMSE: ' if 'RMSE' in line
                                                                          else f'Train {metric}: ')
                            train_valid = float(train_valid)
                            test = float(test.strip())

                            df.loc[f"{dataset_name}_{subset}_{num_fut}",
                                   [f'Train_{metric}', f'Valid_{metric}']] = train_valid
                            df.loc[f"{dataset_name}_{subset}_{num_fut}", f'Test_{metric}'] = test
                    else:
                        if not args.ignore_incomplete:
                            raise ValueError(f'Could not parse {file}: unknown structure.')

                elif model == 'mTAN':
                    metrics = metrics[-3].split(', ')

                    if len(metrics) > 5 and 'train_mse' in metrics[4]:
                        train_mse = float(metrics[4].lstrip('train_mse: '))
                        train_mae = float(metrics[5].lstrip('train_mae: '))
                        train_r2 = float(metrics[6].lstrip('train_r2: '))
                        df.loc[f"{dataset_name}_{subset}_{num_fut}", ['Train_MSE', 'Train_MAE', 'Train_R2']] = \
                            [train_mse, train_mae, train_r2]
                        val_mse = float(metrics[9].lstrip('val_mse: '))
                        val_mae = float(metrics[10].lstrip('val_mae: '))
                        val_r2 = float(metrics[11].lstrip('val_r2: '))
                        df.loc[f"{dataset_name}_{subset}_{num_fut}", ['Valid_MSE', 'Valid_MAE', 'Valid_R2']] = \
                            [val_mse, val_mae, val_r2]
                        test_mse = float(metrics[13].lstrip('test_mse: '))
                        test_mae = float(metrics[14].lstrip('test_mae: '))
                        test_r2 = float(metrics[15].lstrip('test_r2: '))
                        df.loc[f"{dataset_name}_{subset}_{num_fut}", ['Test_MSE', 'Test_MAE', 'Test_R2']] = \
                            [test_mse, test_mae, test_r2]

                    else:
                        if not args.ignore_incomplete:
                            raise ValueError(f'Could not parse {file}: unknown structure.')

                else:
                    raise ValueError(f'Unknown model {model}.')

    # maybe superfluous
    df_dict['ISTS'] = df


def main():
    # models = ['GRU-D', 'CRU', 'mTAN', 'ISTS']

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

    for model, df in df_dict.items():
        df.rename_axis(index='Subset', inplace=True)
        df.to_csv(f'{model}_results.csv')

    if not loss_df.empty:
        loss_df.rename_axis(index='Subset', inplace=True)
        loss_df.to_csv('ists_losses.csv')


if __name__ == '__main__':
    main()
