import argparse
import multiprocessing
import os
import socket
from concurrent import futures
from time import sleep
from pynvml import nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex
import torch

parser = argparse.ArgumentParser('FDB')
# train configs
parser.add_argument('-m', '--model', type=str, nargs='+',
                    help="List of models to train and test separated by spaces.", default=['GRU-D', 'CRU', 'mTAN'])
parser.add_argument('-d', '--datasets_path', type=str, required=True, nargs='+',
                    help="Folder containing pickle datasets or a (list of) path(s) to a pickle dataset.")
parser.add_argument('--device', nargs='+', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                    type=str, help='Device to use for training and testing.')
parser.add_argument('--num_workers', type=str, default=None,
                    help='Number of workers to use for parallelizing models.')
parser.add_argument('--recycle_gpu', action='store_true', default=False,
                    help="Run multiple models on the same GPU to exploit the available VRAM.")

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

vram_usage = {
    'GRU-D': 0,
    'CRU': 3221225472,  # 3 GB in Bytes (CRU uses 3 GB of VRAM)
    'mTAN': 40802189312  # 38 GB in Bytes (mTAN uses 38 GB of VRAM)
}

args = parser.parse_args()


def launch_model(model: str, dataset: str, device_list: dict[str: multiprocessing.Lock],
                 recycle_gpu: bool = False):
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
            if recycle_gpu:
                for dev, locks in device_list.items():
                    if acquired_lock is not None:
                        break

                    for lock in locks:
                        if lock.acquire(blocking=False):
                            print(f'{os.getpid()}: Acquired lock for {dev}')
                            acquired_lock = lock
                            device = dev
                            break

            else:
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

    if acquired_lock is not None:
        acquired_lock.release()


def check_launch_model(model, dataset):
    file_start_map = {
        'GRU-D': 'grud_output_',
        'CRU': 'cru_output_',
        'mTAN': 'mtan_output_'
    }

    launch = True
    completed = False
    log_files = [file for file in os.listdir(wdirs[model]) if file.endswith('.pickle.txt')]

    ran = False
    for file in log_files:
        if dataset.split('/')[-1] in file:
            ran = True
            break

    if ran:
        for file in log_files:
            if file.startswith(file_start_map[model]):
                if dataset.split('/')[-1] in file:
                    with open(os.path.join(wdirs[model], file)) as dataset_file:
                        if model == 'CRU':
                            last_lines = dataset_file.readlines()[-4:]

                            if len(last_lines) < 4:
                                break

                            # check if model crashed, if so, skip execution
                            crashed = "ValueError: NaN in gradient" in last_lines[-1]
                            # be completely sure that the results are structured as they should to say that the run was
                            # successful
                            completed = last_lines[0].startswith('Train R2:') and \
                                last_lines[1].startswith('Train MSE:') and \
                                last_lines[2].startswith('Train MAE:') and last_lines[3].startswith('Duration:')

                            if crashed or completed:
                                print(f"Log file {file}, last lines:")
                                print(last_lines)
                                print("Skipping execution.")
                                launch = False

                                break

                        # TODO: to implement if necessary
                        elif model == 'GRU-D':
                            pass

                        elif model == 'mTAN':
                            pass

    if launch:
        print(f"Execution check True for {model} on {dataset}.")

    return launch, True if completed else False


def main():
    batch_run = False

    if not args.model:
        raise RuntimeError('No models specified.')

    if not args.datasets_path:
        raise RuntimeError('No datasets path specified.')

    models = args.model

    if len(args.datasets_path) == 1:
        args.datasets_path = args.datasets_path[0]

    if os.path.isdir(args.datasets_path):
        batch_run = True
        datasets = [os.path.join(args.datasets_path, file) for file in os.listdir(args.datasets_path)
                    if file.endswith('.pickle')]

    elif os.path.isfile(args.datasets_path):
        datasets = [args.datasets_path]
    else:
        raise ValueError(f'Could not find {args.datasets_path}.')

    if not models:
        raise ValueError('No models specified.')

    if not datasets:
        raise ValueError('No datasets found.')

    with multiprocessing.Manager() as manager:
        if not args.num_workers:
            if args.device == ['all']:
                if models == ['GRU-D']:
                    if multiprocessing.cpu_count() >= 128:
                        max_workers = multiprocessing.cpu_count() // 16
                    else:
                        max_workers = multiprocessing.cpu_count() // 4

                    devices = manager.dict({f'cpu:{idx}': manager.Lock() for idx in range(max_workers)})

                else:
                    devices = manager.dict({f'cuda:{idx}': manager.Lock() for idx in range(torch.cuda.device_count())})

            elif args.recycle_gpu:
                if len(models) > 1:
                    raise RuntimeError('Cannot recycle GPUs when training different models at once.')

                # total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
                device_idx = int(args.device[0].split(':')[-1])
                nvmlInit()
                info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device_idx))
                free_gpu_mem = info.free
                # runnable_models = total_gpu_mem // vram_usage[models[0]]
                runnable_models = free_gpu_mem // vram_usage[models[0]]

                devices = manager.dict()

                for cuda_dev in args.device:
                    locks = manager.list()
                    for _ in range(runnable_models):
                        locks.append(manager.Lock())

                    devices[cuda_dev] = locks
                    # devices = manager.dict({cuda_dev: [manager.Lock()] * runnable_models
                    #                         for cuda_dev in args.device})

                max_workers = len(devices) * runnable_models if models != ['GRU-D'] \
                    else multiprocessing.cpu_count() // 4

            else:
                devices = manager.dict({cuda_dev: manager.Lock() for cuda_dev in args.device})

            if not args.recycle_gpu:
                if models != ['GRU-D']:
                    max_workers = len(devices)
                else:
                    if multiprocessing.cpu_count() >= 128:
                        max_workers = multiprocessing.cpu_count() // 16
                    else:
                        max_workers = multiprocessing.cpu_count() // 4

        else:
            if args.num_workers == 'auto' and args.device == ['cpu']:
                if multiprocessing.cpu_count() >= 128:
                    max_workers = multiprocessing.cpu_count() // 16
                else:
                    max_workers = multiprocessing.cpu_count() // 4

            else:
                max_workers = int(args.num_workers)

            devices = manager.dict({cpu_core: manager.Lock()} for cpu_core in range(max_workers))

        crashed_runs = successful_runs = 0

        # TODO: it appears that the number of processes able to run is somehow limited by the Manager. Having
        #  max_workers > len(devices) still causes the program to run with at most len(devices) processes.
        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for dataset in datasets:
                for model in models:
                    futures_ = list()
                    if batch_run:
                        launch, completed = check_launch_model(model, dataset)

                        if completed:
                            successful_runs += 1
                        else:
                            crashed_runs += 1

                    if (batch_run and launch) or not batch_run:
                        futures_.append(executor.submit(launch_model, model, dataset, devices, args.recycle_gpu))

            if isinstance(datasets, list):
                print("Total datasets found: ", len(datasets))
                print("Recap of already computed datasets:")
                print(f"Total successful runs: {successful_runs}")
                print(f"Total crashed or yet to execute runs: {crashed_runs}", flush=True)

            print("Starting experiments.", flush=True)
            done, not_done = futures.wait(futures_, return_when=futures.ALL_COMPLETED)

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
