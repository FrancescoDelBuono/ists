import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser('FDB Parser')
# train configs
parser.add_argument('--ignore-incomplete', action='store_true', default=False,
                    help="List of models to train and test separated by spaces.")

args = parser.parse_args()


def main():
    models = ['GRU-D', 'CRU', 'mTAN']
    folders = {
        'GRU-D': 'GRU-D',
        'CRU': 'Continuous-Recurrent-Units',
        'mTAN': 'mTAN'
    }
    """folders = {
        'GRU-D': '../GRU-D',
        'CRU': '.',
        'mTAN': '../mTAN'
    }"""
    file_start = {
        'GRU-D': 'grud_output_',
        'CRU': 'cru_output_',
        'mTAN': 'mtan_output_'
    }

    df_dict = {model: pd.DataFrame() for model in models}

    wdir = os.getcwd()

    for model in models:
        df = df_dict[model]
        os.chdir(folders[model])
        for file in os.listdir():
            if file.startswith(file_start[model]):
                print(f'Parsing {file}...')
                dataset_name = file.lstrip(file_start[model]).rstrip('.pickle.txt')
                with open(file, 'r') as log_file:
                    metrics = log_file.readlines()[-20:]

                    if 'ValueError' in metrics[-1]:  # if error save NaN
                        for mode in ['Train', 'Valid', 'Test']:
                            df.loc[dataset_name, [f'{mode}_R2', f'{mode}_MSE', f'{mode}_MAE']] = np.nan

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

                                    if type(metrics) == list:
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

                                    train_valid = train_valid.rstrip(', ').lstrip(f'Train RMSE: ' if 'RMSE' in line
                                                                                  else f'Train {metric}: ')
                                    train_valid = float(train_valid)
                                    test = float(test.strip())

                                    df.loc[dataset_name, [f'Train_{metric}', f'Valid_{metric}']] = train_valid
                                    df.loc[dataset_name, f'Test_{metric}'] = test
                            else:
                                if not args.ignore_incomplete:
                                    raise ValueError(f'Could not parse {file}: unknown structure.')

                        elif model == 'mTAN':
                            metrics = metrics[-3].split(', ')

                            if len(metrics) > 5 and 'train_mse' in metrics[4]:
                                train_mse = float(metrics[4].lstrip('train_mse: '))
                                train_mae = float(metrics[5].lstrip('train_mae: '))
                                train_r2 = float(metrics[6].lstrip('train_r2: '))
                                df.loc[dataset_name, ['Train_MSE', 'Train_MAE', 'Train_R2']] = \
                                    [train_mse, train_mae, train_r2]
                                val_mse = float(metrics[9].lstrip('val_mse: '))
                                val_mae = float(metrics[10].lstrip('val_mae: '))
                                val_r2 = float(metrics[11].lstrip('val_r2: '))
                                df.loc[dataset_name, ['Valid_MSE', 'Valid_MAE', 'Valid_R2']] = \
                                    [val_mse, val_mae, val_r2]
                                test_mse = float(metrics[13].lstrip('test_mse: '))
                                test_mae = float(metrics[14].lstrip('test_mae: '))
                                test_r2 = float(metrics[15].lstrip('test_r2: '))
                                df.loc[dataset_name, ['Test_MSE', 'Test_MAE', 'Test_R2']] = \
                                    [test_mse, test_mae, test_r2]

                            else:
                                if not args.ignore_incomplete:
                                    raise ValueError(f'Could not parse {file}: unknown structure.')

                        else:
                            raise ValueError(f'Unknown model {model}.')

        os.chdir(wdir)

    for model, df in df_dict.items():
        df.to_csv(f'{model}_results.csv')


if __name__ == '__main__':
    main()
