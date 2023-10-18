import argparse
import multiprocessing
import os
import socket
from concurrent import futures
from time import sleep

import torch

parser = argparse.ArgumentParser('FDB')
# train configs
parser.add_argument('-m', '--model', type=str, nargs='+',
                    help="List of models to train and test separated by spaces.", default=['GRU-D', 'CRU', 'mTAN'])
parser.add_argument('-d', '--datasets_path', type=str, required=True, nargs='+',
                    help="Folder containing pickle datasets or a (list of) path(s) to a pickle dataset.")
parser.add_argument('--device', nargs='+', default='cuda:0',
                    type=str, help='Device to use for training and testing.')

hostname = socket.gethostname()

if 'gnode' in hostname or 'cnode' in hostname or 'fnode' in hostname:  # ARIES
    home_path = '/unimore_home/gguiduzzi'

elif 'fpdgx1' in socket.gethostname():  # LYRA
    home_path = '/trafair/gguiduzzi'

else:  # SPARC20
    home_path = '/home/giacomo.guiduzzi'

interpreters = {
    'GRU-D': f'{home_path}/.virtualenvs/GRU-D/bin/python',
    'CRU': f'{home_path}/.virtualenvs/cru_sm80/bin/python',
    'mTAN': f'{home_path}/.virtualenvs/mtan_sm80/bin/python'
}

wdirs = {
    'GRU-D': f'{home_path}/GRU-D',
    'CRU': f'{home_path}/Continuous-Recurrent-Units',
    'mTAN': f'{home_path}/mTAN'
}

scripts = {
    'GRU-D': 'main.py',
    'CRU': 'run_experiment.py',
    'mTAN': 'src/tan_forecasting.py'
}

parameters = {
    'GRU-D': 'fdb {} 2>&1 | tee grud_output_{}.txt',
    'CRU': '--dataset fdb --task forecast -lsd 30 --epochs 10 --sample-rate 0.5 --filename {} --batch-size 64 '
           '--device {} 2>&1 | tee cru_output_{}.txt',
    'mTAN': '--alpha 100 --niters 10 --lr 0.0001 --batch-size 64 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 '
            '--enc mtan_rnn --dec mtan_rnn --save 1 --norm --kl --learn-emb --k-iwae 1 --dataset fdb '
            '--filename {} --device {} --normalize_tp 2>&1 | tee mtan_output_{}.txt'
}

args = parser.parse_args()


def launch_model(model, dataset, device_list):
    device = 'cpu'
    acquired_lock = None

    print(f"Process {os.getpid()} started with parameters: {model}, {dataset}")

    if '/' in dataset:
        dataset_name = dataset.split('/')[-1]

    elif '\\' in dataset:
        dataset_name = dataset.split('\\')[-1]
    else:
        dataset_name = dataset

    if model != 'GRUD-D':  # GRU-D must run on CPU because of tensorflow compatibility problems with CUDA
        while True:
            for dev, lock in device_list.items():
                if lock.acquire(blocking=False):
                    print(f'{os.getpid()}: Acquired lock for {dev}')
                    acquired_lock = lock
                    device = dev
                    break

            if acquired_lock:
                break
            else:
                # couldn't find a free lock, waiting
                print(f'{os.getpid()}: Waiting for a free lock...')
                sleep(5)

    interpreter = interpreters[model]
    wdir = wdirs[model]
    os.chdir(wdir)
    script = scripts[model]
    if model == 'GRU-D':
        cmd_parameters = parameters[model].format(dataset, dataset_name)
    elif model == 'CRU':
        cmd_parameters = parameters[model].format(dataset, device, dataset_name)
    elif model == 'mTAN':
        cmd_parameters = parameters[model].format(dataset, device, dataset_name)
    else:
        raise RuntimeError(f'Unknown model {model}')

    # command = interpreter + ' ' + wdir + '/' + script + ' ' + parameter
    command = ' '.join([interpreter, script, cmd_parameters])

    print(f'{os.getpid()}: Launching {model} using {dataset_name} on {device}')
    os.system(command)
    print(f'{os.getpid()}: Finished {model} using {dataset_name} on {device}')

    acquired_lock.release()


def main():
    if not args.model:
        raise RuntimeError('No models specified.')

    if not args.datasets_path:
        raise RuntimeError('No datasets path specified.')

    models = args.model

    if len(args.datasets_path) == 1:
        args.datasets_path = args.datasets_path[0]

    if type(args.datasets_path) == str:
        if os.path.isdir(args.datasets_path):
            datasets = [file for file in os.listdir(args.datasets_path) if file.endswith('.pickle')]
        elif os.path.isfile(args.datasets_path):
            datasets = [args.datasets_path]
        else:
            raise ValueError(f'Could not find {args.datasets_path}.')

    else:
        if not all([file.endswith('.pickle') for file in os.listdir(args.datasets_path)]):
            raise ValueError('Not all files in the specified folder are pickle files.')
        else:
            datasets = args.datasets_path

    if not models:
        raise ValueError('No models specified.')

    if not datasets:
        raise ValueError('No datasets found.')

    with multiprocessing.Manager() as manager:
        if args.device == 'all':
            devices = manager.dict({f'cuda:{idx}': manager.Lock() for idx in range(torch.cuda.device_count())})
        else:
            devices = manager.dict({cuda_dev: manager.Lock() for cuda_dev in args.device})

        with futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
            for dataset in datasets:
                for model in models:
                    futures_ = list()

                    futures_.append(executor.submit(launch_model, model, dataset, devices))

            done, not_done = futures.wait(futures_, return_when=futures.FIRST_EXCEPTION)

            futures_exceptions = [future.exception() for future in done]
            failed_futures = sum(map(lambda exception_: True if exception_ is not None else False,
                                     futures_exceptions))

            if failed_futures > 0:
                print("Could not train and evaluate all models. Thrown exceptions: ")

                for exception in futures_exceptions:
                    print(exception)

                raise RuntimeError(f"Couldn't train and evaluate all models, {failed_futures} processes failed.")

            if failed_futures == 0:
                print("Experiments concluded successfully.")


if __name__ == '__main__':
    main()
