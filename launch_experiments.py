import argparse
import multiprocessing
import os
from concurrent import futures
from time import sleep

import torch

parser = argparse.ArgumentParser('FDB')
# train configs
parser.add_argument('models', type=str, help="List of models to train and test separated by spaces.")
parser.add_argument('datasets_path', type=str, help="Path of folder containing pickle datasets.")

interpreters = {
    'GRU-D': '/trafair/gguiduzzi/.virtualenvs/GRU-D/bin/python',
    'CRU': '/trafair/gguiduzzi/.virtualenvs/cru_sm80/bin/python',
    'mTAN': '/trafair/gguiduzzi/.virtualenvs/mtan_sm80/bin/python'
}

wdirs = {
    'GRU-D': '/trafair/gguiduzzi/GRU-D',
    'CRU': '/trafair/gguiduzzi/Continuous-Recurrent-Units',
    'mTAN': '/trafair/gguiduzzi/mTAN'
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
        cmd_parameters = parameters[model].format(dataset, dataset)
    elif model == 'CRU':
        cmd_parameters = parameters[model].format(dataset, device, dataset)
    elif model == 'mTAN':
        cmd_parameters = parameters[model].format(dataset, device, dataset)
    else:
        raise RuntimeError(f'Unknown model {model}')

    # command = interpreter + ' ' + wdir + '/' + script + ' ' + parameter
    command = ' '.join([interpreter, script, cmd_parameters])

    print(f'{os.getpid()}: Launching {model} using {dataset} on {device}')
    os.system(command)
    print(f'{os.getpid()}: Finished {model} using {dataset} on {device}')

    acquired_lock.release()


def main():
    if not args.models:
        raise RuntimeError('No models specified.')

    if not args.datasets_path:
        raise RuntimeError('No datasets path specified.')

    models = args.models.split()
    datasets = [file for file in os.listdir(args.datasets_path) if file.endswith('.pickle')]

    if not models:
        raise ValueError('No models specified.')

    if not datasets:
        raise ValueError('No datasets found.')

    with multiprocessing.Manager() as manager:
        devices = manager.dict({f'cuda:{idx}': manager.Lock() for idx in range(torch.cuda.device_count())})

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
