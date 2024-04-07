import argparse
import multiprocessing
import os
import socket
import subprocess
import time
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

parser.add_argument('--force_execution', action='store_true', default=False,
                    help="Force the execution of the models even if the results are already present.")

parser.add_argument('--scaler', type=str, default=['standard'], nargs='+',
                    help="List of scalers to use for preprocessing the data, one for each model.")

parser.add_argument('--short_run', action='store_true', default=False,
                    help="Run a short version of the experiments for debugging purposes.")

args = parser.parse_args()

if args.scaler[0].lower() == 'none':
    args.scaler = None

if args.scaler:
    if len(args.scaler) == len(args.model):
        scaler_map = {model: scaler for model, scaler in zip(args.model, args.scaler)}
    elif len(args.scaler) == 1:
        scaler_map = {model: args.scaler[0] for model in args.model}
    else:
        raise ValueError(f"Invalid number of scalers specified: "
                         f"{len(args.scaler)} scalers for {len(args.model)} models.")

else:
    scaler_map = {model: 'None' for model in args.model}

hostname = socket.gethostname()

if 'gnode' in hostname or 'cnode' in hostname or 'fnode' in hostname:  # ARIES
    home_path = '/unimore_home/gguiduzzi'
    models_path = ''

elif 'fpdgx1' in socket.gethostname():  # LYRA
    home_path = '/trafair/gguiduzzi'
    models_path = ''

else:  # SPARC20
    home_path = '/home/giacomo.guiduzzi'
    models_path = os.path.join(home_path, 'softlab_sharepoint', 'Projects', 'Adbpo', 'Models')

if 'sparc20' in hostname:
    interpreters = {
        'GRU-D': os.path.join(home_path, '.virtualenvs', 'GRU-D', 'bin', 'python'),
        'CRU': os.path.join(home_path, '.virtualenvs', 'Continuous-Recurrent-Units', 'bin', 'python'),
        'mTAN': os.path.join(home_path, '.virtualenvs', 'mTAN', 'bin', 'python')
    }
else:
    interpreters = {
        'GRU-D': os.path.join(home_path, '.virtualenvs', 'GRU-D', 'bin', 'python'),
        'CRU': os.path.join(home_path, '.virtualenvs', 'cru_sm80', 'bin', 'python'),
        'mTAN': os.path.join(home_path, '.virtualenvs', 'mtan_sm80', 'bin', 'python')
    }

wdirs = {
    'GRU-D': os.path.join(home_path, models_path, 'GRU-D'),
    'CRU': os.path.join(home_path, models_path, 'Continuous-Recurrent-Units'),
    'mTAN': os.path.join(home_path, models_path, 'mTAN'),
}

scripts = {
    'GRU-D': 'main.py',
    'CRU': 'run_experiment.py',
    'mTAN': 'src/tan_forecasting.py'
}

parameters = {
    'GRU-D': 'fdb {} --scaler {} {}',
    'CRU': '--dataset fdb --task forecast -lsd 30 --epochs 10 --sample-rate 0.5 --filename {} --batch-size 64 '
           '--device {} --scaler {} {}',
    'mTAN': '--alpha 100 --niters 10 --lr 0.0001 --batch-size 64 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 '
            '--enc mtan_rnn --dec mtan_rnn --save 1 --norm --kl --learn-emb --k-iwae 1 --dataset fdb '
            '--filename {} --device {} --scaler {} --normalize_tp {}'
}

output_files = {
    'GRU-D': 'grud_output_{}_{}.txt',
    'CRU': 'cru_output_{}_{}.txt',
    'mTAN': 'mtan_output_{}_{}.txt'
}

short_run_parameters = {
    'GRU-D': '--epochs 1 --batches 100',
    'CRU': '--epochs 1 --batches 100',
    'mTAN': '--niters 1 --batches 100'
}

vram_usage = {
    'GRU-D': 0,
    'CRU': 3221225472,  # 3 GB in Bytes (CRU uses 3 GB of VRAM)
    'mTAN': 40802189312  # 38 GB in Bytes (mTAN uses 38 GB of VRAM)
}


def launch_model(model: str, dataset: str, device_list: dict[str: multiprocessing.RLock],
                 recycle_gpu: bool = False):
    device = 'cpu'
    acquired_lock = None

    print(f"Process {os.getpid()} started with parameters: {model}, {dataset}")

    if os.sep in dataset:
        dataset_name = dataset.split(os.sep)[-1]

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
        cmd_parameters = parameters[model].format(dataset,
                                                  scaler_map[model],
                                                  short_run_parameters[model] if args.short_run else '')
    elif model == 'CRU':
        cmd_parameters = parameters[model].format(dataset,
                                                  device,
                                                  scaler_map[model],
                                                  short_run_parameters[model] if args.short_run else '')
    elif model == 'mTAN':
        cmd_parameters = parameters[model].format(dataset,
                                                  device,
                                                  scaler_map[model],
                                                  short_run_parameters[model] if args.short_run else '')
    else:
        raise RuntimeError(f'Unknown model {model}')

    # command = interpreter + ' ' + wdir + '/' + script + ' ' + parameter
    command = ' '.join([interpreter, script, cmd_parameters])
    output_file = output_files[model].format(scaler_map[model], dataset_name)

    one_day_timeout = 60 * 60 * 24  # 24 hours in seconds

    try:
        print(f'{os.getpid()}: Launching {model} using {dataset_name} on {device}')
        # os.system(command)
        start_time = time.time()
        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        with open(os.path.join(wdir, output_file), 'w') as out_file:
            while True:
                result = proc.poll()
                if result is not None:
                    subprocess_exit_code = result
                    break
                else:
                    output = proc.stdout.readline()
                    line_counter = 0
                    while output:
                        out_file.write(output)
                        print(output, end='', flush=True)

                        # this mechanism is to avoid that processes like GRU-D keep this process busy reading
                        # without ever checking the timeout threshold.
                        line_counter += 1
                        if line_counter > 1000:
                            break

                        output = proc.stdout.readline()

                if time.time() - start_time > one_day_timeout:
                    print(f"{os.getpid()}: Timeout reached, sending SIGINT to process {proc.pid} "
                          f"(model {model} using {dataset_name} on {device}).")

                    proc.terminate()  # Try polite termination first

                    try:
                        print(f"Waiting one minute for process {proc.pid} to gracefully exit...")
                        proc.wait(timeout=60)  # Wait for process to gracefully exit
                        subprocess_exit_code = proc.returncode

                    except subprocess.TimeoutExpired:
                        print(f"Process {proc.pid} did not exit gracefully, sending SIGTERM.")
                        proc.kill()  # Forceful termination if unresponsive
                        subprocess_exit_code = -9

                    break

                sleep(2)  # sleep a bit to avoid busy waiting

            output = proc.stdout.read()
            # write to log all the output that was still in the buffer
            if output:
                out_file.write(output)
                print(output, end='', flush=True)

        print(f'{os.getpid()}: Finished {model} using {dataset_name} on {device}')

    except Exception as e:
        print(f"Error while running {model} on {dataset_name} on {device}.")
        raise e

    finally:
        if acquired_lock:
            acquired_lock.release()

    return subprocess_exit_code


def check_launch_model(model: str, dataset: str) -> tuple[bool, bool]:
    """
    Scans the available logs in the model directory to check if the model has already been executed on the dataset.
    If the model has already been executed on that dataset, the function checks if the execution was successful
    or if the model crashed. If the model crashed, the function returns True, else False.
    If the model has already been executed and the run was successful, the function returns False.
    The second return argument is True if the model has already been executed and the run was successful, else False.
    This argument has the sole purpose of computing statistics.
    """

    def check_correctness(log_file) -> tuple[bool, bool]:
        with (open(os.path.join(wdirs[model], log_file)) as dataset_log_file):

            # TODO: manage timeout: tail -n 1 *.pickle.txt | grep -v "Duration: *"

            last_lines = dataset_log_file.readlines()[-5:]

            if len(last_lines) < 4:
                errors_in_log = any(['Timeout' in line or
                                     'ValueError' in line or
                                     'FileNotFound' in line
                                     for line in last_lines])

                completed = False

                if not errors_in_log:  # search for a process that is still running with this dataset
                    cmd = f"ps aux | grep {interpreters[model]}.*{dataset.split('/')[-1]}"
                    print(f"Log file {log_file} is too short, searching for a process with the same dataset using:")
                    print(cmd)
                    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

                    if process.stderr:
                        print(f"Error while running command {cmd}: {process.stderr}")
                        launch = True

                    else:
                        found = False
                        for line in process.stdout:
                            if interpreters[model] in line and dataset.split('/')[-1] in line:
                                print(f"Found a process still running with the same dataset: {line}")
                                launch = False
                                found = True
                                break

                        if not found:
                            print("No process found, re-executing.")
                            launch = True

                else:
                    print(f"Found errors in log file {log_file}. Last lines:")
                    print(last_lines)
                    launch = True

                return launch, completed

            if model == 'CRU':
                print("Checking CRU log file.")
                '''
                Train R2: 0.8168, Test R2: 0.7675
                Train MSE: 0.0038, Test MSE: 0.0054
                Train MAE: 0.0482, Test MAE: 0.0576
                Duration: 2 days, 17:52:54.155161
                '''

                # check if the model crashed, if so, skip execution
                # need to be completely sure that the results are structured as they should to say
                # that the run was successful
                completed = last_lines[-4].startswith('Train R2:') and \
                            last_lines[-3].startswith('Train MSE:') and \
                            last_lines[-2].startswith('Train MAE:') and \
                            last_lines[-1].startswith('Duration:')

                print(f"Log file {log_file}, last lines:")
                print(last_lines)

                if completed:
                    print("Skipping execution.")
                    launch = False

                else:
                    print("Model crashed, re-executing.")
                    launch = True

            elif model == 'GRU-D':
                print("Checking GRU-D log file.")
                '''
                Performance metrics: 
                R2 score:  [-0.20752941465679964, -0.10023247384247913, -0.3530475548581993]
                MAE score:  [0.12207800459720515, 0.11395531709604426, 0.1395537585637083]
                MSE score:  [0.025358614210495553, 0.022113583850112563, 0.03154178228352191]
                ====================
                '''
                completed = last_lines[-5].startswith('Performance metrics:') and \
                            last_lines[-4].startswith('R2 score:') and \
                            last_lines[-3].startswith('MAE score:') and \
                            last_lines[-2].startswith('MSE score:')
                # last_lines[-1] is the ===== line

                print(f"Log file {log_file}, last lines:")
                print(last_lines)

                if completed:
                    print("Skipping execution.")
                    launch = False
                else:
                    print("Model crashed, re-executing.")
                    launch = True

            elif model == 'mTAN':
                print("Checking mTAN log file.")
                '''
                (Iter: 10, recon_loss: 27.0015, mse_loss: 0.0068, acc: 0.0010, train_mse: 0.0001, train_mae: 
                0.0010, train_r2: 0.0104, val_loss: 0.0065, val_acc: 0.0623, val_mse: 0.0065, val_mae: 
                0.0623, val_r2: 0.6894, test_acc: 0.0968, test_mse: 0.0150, test_mae: 0.0968, 
                test_r2: 0.3574) (same line)
                Best val loss: 0.006492583523961666
                Total time: 1228.6564812660217
                '''
                completed = last_lines[-3].startswith('Iter: 10') and \
                            'test_mse' in last_lines[-3] and \
                            'test_mae' in last_lines[-3] and \
                            'test_r2' in last_lines[-3] and \
                            last_lines[-2].startswith('Best val loss:') and \
                            last_lines[-1].startswith('Total time:')

                print(f"Log file {log_file}, last lines:")
                print(last_lines)

                if completed:
                    print("Skipping execution.")
                    launch = False
                else:
                    print("Model crashed, re-executing.")
                    launch = True

            return launch, completed

    log_filename = output_files[model].format(scaler_map[model], dataset.split('/')[-1])

    to_launch = True
    has_completed = False
    # log file name example: cru_output_minmax_french_th18_0_nan0_nf7.pickle.txt

    if log_filename in os.listdir(wdirs[model]):
        # the model was run before
        to_launch, has_completed = check_correctness(log_filename)

    # if the model wasn't run before, the launch and completed flags are not changed
    if to_launch:
        print(f"Execution check True for {model} on {dataset}.")

    return to_launch, has_completed


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

                    devices = manager.dict({f'cpu:{idx}': manager.RLock() for idx in range(max_workers)})

                else:
                    devices = manager.dict({f'cuda:{idx}': manager.RLock() for idx in range(torch.cuda.device_count())})

            elif args.recycle_gpu:
                if len(models) > 1:
                    raise RuntimeError('Cannot recycle GPUs when training different models at once.')

                print("Recycling GPUs.")

                # total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
                device_idx = int(args.device[0].split(':')[-1])
                nvmlInit()
                info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(device_idx))
                free_gpu_mem = info.free
                # runnable_models = total_gpu_mem // vram_usage[models[0]]
                runnable_models = free_gpu_mem // vram_usage[models[0]]

                print("Runnable models:", runnable_models)

                devices = manager.dict()

                for cuda_dev in args.device:
                    locks = manager.list()
                    for _ in range(runnable_models):
                        locks.append(manager.RLock())

                    devices[cuda_dev] = locks
                    # devices = manager.dict({cuda_dev: [manager.RLock()] * runnable_models
                    #                         for cuda_dev in args.device})

                max_workers = len(devices) * runnable_models if models != ['GRU-D'] \
                    else multiprocessing.cpu_count() // 4

            else:  # devices != ['all'] and not recycle_gpu
                devices = manager.dict({cuda_dev: manager.RLock() for cuda_dev in args.device})

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

            devices = manager.dict({cpu_core: manager.RLock()} for cpu_core in range(max_workers))

        crashed_runs = successful_runs = 0

        print("Available devices: ", devices.keys())
        print("Num workers: ", max_workers)

        # TODO: it appears that the number of processes able to run is somehow limited by the Manager. Having
        #  max_workers > len(devices) still causes the program to run with at most len(devices) processes.

        # TODO: maybe use an init function (argument to ProcessPoolExecutor) to start the
        #  process only if the RAM is sufficient (else sleep for like 5 minutes).
        #  This requires a way to estimate the memory footprint of models and their datasets in advance.
        #  (CRU is problematic regarding memory footprint)
        #  RAPIDS RMM library can be used to manage memory allocation and deallocation in CUDA, could be useful.

        # The usage of max_tasks_per_child implies mp_context = 'spawn'. The parameter is available only from
        # Python 3.11
        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for dataset in datasets:
                for model in models:
                    futures_ = list()
                    if batch_run:
                        if args.force_execution:
                            launch = True
                            completed = False
                        else:
                            launch, completed = check_launch_model(model, dataset)

                        if completed:
                            successful_runs += 1
                        else:
                            crashed_runs += 1

                    if (batch_run and launch) or not batch_run:
                        futures_.append(executor.submit(launch_model, model, dataset, devices, args.recycle_gpu))

            if isinstance(datasets, list):
                print("Total datasets found: ", len(datasets))
                if not args.force_execution:
                    print("Recap of already computed datasets:")
                    print(f"Total successful runs: {successful_runs}")
                    print(f"Total crashed or yet to execute runs: {crashed_runs}", flush=True)

                else:
                    print("Forcing execution on all datasets.")

                print("Starting experiments.", flush=True)

                done, not_done = futures.wait(futures_, return_when=futures.ALL_COMPLETED)

        futures_exceptions = [future.exception() for future in done]
        failed_futures = sum(map(lambda exception_: True if exception_ is not None else False,
                                 futures_exceptions))

        if not_done:
            print(f"{len(not_done)} processes timed out.")

        if failed_futures > 0:
            print("Could not train and evaluate all models. Thrown exceptions: ")

            for future_, exception in futures_exceptions:
                print(exception)

            raise RuntimeError(f"Couldn't train and evaluate all models, {failed_futures} processes failed.")

        if failed_futures == 0:
            print("Experiments concluded successfully.")


if __name__ == '__main__':
    main()
