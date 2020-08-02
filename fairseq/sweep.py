import argparse
from collections import OrderedDict
import datetime
import itertools
import os
import random
import shlex
import shutil
import subprocess


def get_args():
    parser = argparse.ArgumentParser('Script for launching hyperparameter sweeps')
    parser.add_argument('-d', '--data', required=True, help='path to data directory')
    parser.add_argument('-p', '--prefix', required=True,
                        help='save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>')
    parser.add_argument('-t', '--num-trials', required=True, type=int,
                        help='number of random hyperparam configurations to try (-1 for grid search)')
    parser.add_argument('-g', '--num-gpus', type=int, required=True, help='number of GPUs per node')
    parser.add_argument('-n', '--num-nodes', type=int, default=1, help='number of nodes for distributed training')
    parser.add_argument('--mem', '--mem', help='memory to request')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--baseline-model', help='path to baseline model from which to resume training')
    parser.add_argument('--checkpoints-dir',
                        default=os.path.join('/checkpoint', os.environ['USER'], str(datetime.date.today())),
                        help='save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>')
    parser.add_argument('--resume-failed', action='store_true',
                        help='resume any runs that failed (assumes --num-trials and --seed are the same)')
    parser.add_argument('--resume-finished', action='store_true',
                        help='force any runs that finished to begin again (uncommon)')
    parser.add_argument('--dry-run', action='store_true',
                        help='output only a list of actions to perform without performing them')
    parser.add_argument('--local', action='store_true',
                        help='run locally instead of submitting remote job')
    parser.add_argument('--partition', help='partition to run on', default='learnfair')
    parser.add_argument('--reservation', help='reservation to run on')
    parser.add_argument('--exclusive', action='store_true',
                        help='if set, get exclusive host')
    parser.add_argument('--dep', metavar='JOBID', type=int,
                        help='add JOBID as a dependency (i.e., wait for it to finish)')
    parser.add_argument('--sequential', action='store_true',
                        help='schedule jobs to run sequentially')
    parser.add_argument('--time', default='4320',
                        help='expected job duration in minutes')
    args = parser.parse_args()
    return args


class hyperparam(object):
    """Base class for defining hyperparameters."""

    def __init__(self, name, values=None, binary_flag=False, save_dir_key=None):
        """
        Arguments:
        - name : the name of the hyperparameter (e.g., `--dropout`)
        - values : the set of values to sweep over (e.g., `[0.0, 0.1, 0.2]`)
        - binary_flag : whether the hyperparameter uses a boolean flag (e.g., `--no-save`)
        - save_dir_key : function that takes the hyperparameter value and returns the "key"
                         to be appended to the output directory name
        """
        self.name = name
        if values is None:  # syntactic sugar for binary flags
            self.values = [True]
            self.binary_flag = True
        else:
            self.values = values if isinstance(values, list) else [values]
            self.binary_flag = binary_flag
        self.save_dir_key = save_dir_key
        self.current_value = None

        if len(self.values) > 1 and self.save_dir_key is None:
            raise ValueError(f'{name} has more than one value but is missing a save_dir_key!')

    def get_cli_args(self):
        if self.binary_flag:
            return [self.name] if self.current_value else []
        else:
            return [self.name, self.current_value]

    def get_save_dir_key(self):
        if self.save_dir_key is None:
            return None
        if self.binary_flag:
            return self.save_dir_key(1) if self.current_value else None
        return self.save_dir_key(self.current_value)


def main(get_grid, postprocess_hyperparams):
    args = get_args()

    if args.local:
        args.num_nodes = 1

    # compute all possible hyperparameter configurations
    grid = get_grid(args)
    grid_product = list(itertools.product(*[hp.values for hp in grid]))

    # randomly shuffle configurations
    random.seed(args.seed)
    random.shuffle(grid_product)

    for i, hp_values in enumerate(grid_product):
        config = OrderedDict()
        for hp, value in zip(grid, hp_values):
            config[hp.name] = hp
            config[hp.name].current_value = value

        # postprocess hyperparams
        postprocess_hyperparams(args, config)

        # launch training
        job_id = launch_train(args, config)

        if args.sequential and not args.local and job_id is not None:
            args.dep = job_id

        if i == args.num_trials - 1:
            break


def launch_train(args, config):
    def dry_run(msg):
        if args.dry_run:
            print(f'| dry-run:  {msg}')
        return args.dry_run

    # compute save_dir
    save_dir_key = '.'.join(filter(
        lambda save_dir_key: save_dir_key is not None,
        [hp.get_save_dir_key() for hp in config.values()]
    ))
    save_dir_key = save_dir_key.replace(",", "_")
    num_total_gpus = args.num_nodes * args.num_gpus
    save_dir = os.path.join(args.checkpoints_dir, f'{args.prefix}.{save_dir_key}.ngpu{num_total_gpus}')

    # create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        if not dry_run(f'create directory: {save_dir}'):
            os.makedirs(save_dir)

        # copy baseline model
        checkpoint_last = os.path.join(save_dir, 'checkpoint_last.pt')
        if args.baseline_model and not os.path.exists(checkpoint_last) and \
                not dry_run(f'initialize with baseline model: {args.baseline_model}'):
            if not os.path.exists(args.baseline_model):
                raise FileNotFoundError(f'Cannot find baseline model: {args.baseline_model}')
            shutil.copyfile(args.baseline_model, checkpoint_last)

    # check for whether the run failed
    if has_finished(save_dir):
        if args.resume_finished:
            dry_run(f'restart previously finished run: {save_dir}')
        else:
            print(f'skip finished run (override with --resume-finished): {save_dir}')
            return
    elif has_failed(save_dir):
        if args.resume_failed:
            dry_run(f'resume failed run: {save_dir}')
        else:
            print(f'skip failed run (override with --resume-failed): {save_dir}')
            return
    elif has_started(save_dir):
        print(f'skip in progress run: {save_dir}')
        return

    # generate train command
    train_cmd = ['python', 'train.py', args.data, '--save-dir', save_dir]
    if args.num_nodes > 1:
        train_cmd.extend(['--distributed-world-size', str(args.num_nodes * args.num_gpus)])
        train_cmd.extend(['--distributed-port', str(get_random_port())])
    for hp in config.values():
        train_cmd.extend(map(str, hp.get_cli_args()))
    if args.dry_run:
        train_cmd_str = ' '.join(train_cmd)
        dry_run(f'train command: {train_cmd_str}')

    # start training
    if args.local:
        assert args.num_nodes == 1, 'distributed training cannot be combined with --local'
        if not dry_run('start training locally'):
            env = os.environ.copy()
            if 'CUDA_VISIBLE_DEVICES' not in env:
                env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(args.num_gpus)))
            train_proc = subprocess.Popen(train_cmd, env=env)
            train_proc.wait()
    else:
        train_log = os.path.join(save_dir, 'train.log')
        train_stderr = os.path.join(save_dir, 'train.stderr.%j')  # %j = slurm job id

        # build command
        excluded_hosts = os.environ.get('EXCLUDED_HOSTS', None)
        included_hosts = os.environ.get('INCLUDED_HOSTS', None)
        srun_cmd = [
                       'srun',
                       '--job-name', f'{args.prefix}.{save_dir_key}',
                       '--output', train_log,
                       '--error', train_stderr,
                       '--open-mode', 'append',
                       '--unbuffered',
                   ] + train_cmd
        ntasks_per_node = args.num_gpus if args.num_nodes > 1 else 1
        sbatch_cmd = [
            'sbatch',
            '--job-name', f'{args.prefix}.{save_dir_key}',
            '--gres', f'gpu:{args.num_gpus}',
            '--nodes', str(args.num_nodes),
            '--ntasks-per-node', str(ntasks_per_node),
            '--cpus-per-task', str(int(8 * args.num_gpus / ntasks_per_node)),
            '--output', train_log,
            '--error', train_stderr,
            '--open-mode', 'append',
            # '--no-requeue',
            '--signal', 'B:USR1@180',
        ]
        if args.comment:
            sbatch_cmd += ['--comment', args.comment]
        if args.partition:
            sbatch_cmd += ['--partition', args.partition]
        if args.reservation:
            sbatch_cmd += ['--reservation', args.reservation]
        if args.exclusive:
            sbatch_cmd += ['--exclusive']

        if args.dep is not None:
            sbatch_cmd.extend(['-d', str(args.dep)])
        if args.time is not None:
            sbatch_cmd.extend(['--time', args.time])
        if args.mem is not None:
            sbatch_cmd += ['--mem', args.mem]
        else:
            sbatch_cmd += ['--mem-per-cpu', '6G']
        sbatch_cmd += ['-x', excluded_hosts] if excluded_hosts is not None else []
        sbatch_cmd += ['-w', included_hosts] if included_hosts is not None else []

        srun_cmd_str = ' '.join(map(shlex.quote, srun_cmd)) + ' &'

        wrapped_cmd = requeue_support() + '\n' + srun_cmd_str + ' \n wait $! \n sleep 610 & \n wait $!'

        sbatch_cmd += ['--wrap', wrapped_cmd]
        sbatch_cmd_str = ' '.join(map(shlex.quote, sbatch_cmd))

        if args.dry_run:
            dry_run('start remote training')
            dry_run(f'- log stdout to: {train_log}')
            dry_run(f'- log stderr to: {train_stderr}')
            dry_run(f'- run command: {sbatch_cmd_str}')
            sbatch_cmd += ['--test-only']
            with subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE) as train_proc:
                stdout = train_proc.stdout.read().decode('utf-8')
                print(stdout)
        else:
            with open(train_log, 'a') as train_log_h:
                # log most recent git commit
                git_commit = subprocess.check_output(
                    'git log | head -n 1', shell=True, encoding='utf-8')
                print(git_commit.rstrip(), file=train_log_h)
                if args.baseline_model:
                    print(f'baseline model: {args.baseline_model}', file=train_log_h)
                print(f'running command: {sbatch_cmd_str}\n')
                with subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE) as train_proc:
                    stdout = train_proc.stdout.read().decode('utf-8')
                    print(stdout, file=train_log_h)
                    job_id = int(stdout.rstrip().split()[-1])
                    return job_id


def has_finished(save_dir):
    train_log = os.path.join(save_dir, 'train.log')
    if not os.path.exists(train_log):
        return False
    with open(train_log, 'r') as h:
        lines = h.readlines()
        if len(lines) == 0:
            return False
        if 'done training' in lines[-1]:
            return True
    return False


def has_failed(save_dir):
    if not os.path.exists(save_dir):
        return False

    # find max job id
    job_ids = []
    for fn in os.listdir(save_dir):
        if fn.startswith('train.stderr.'):
            job_ids.append(int(fn.split('.')[-1]))
    if len(job_ids) == 0:
        return False
    max_job_id = max(job_ids)

    def _has_failed(stderr_fn):
        with open(stderr_fn, 'r') as h:
            for line in h:
                if len(line.strip()) > 0:
                    # assume that any output in stderr indicates an error
                    return True
        return False

    return _has_failed(os.path.join(save_dir, f'train.stderr.{max_job_id}'))


def has_started(save_dir):
    train_log = os.path.join(save_dir, 'train.log')
    if not os.path.exists(train_log):
        return False
    return True


def get_random_port():
    rng_state = random.getstate()
    random.seed()
    port = random.randint(50000, 60000)
    random.setstate(rng_state)
    return port


def requeue_support():
    return """
        trap_handler () {
           echo "Caught signal: " $1
           # SIGTERM must be bypassed
           if [ "$1" = "TERM" ]; then
               echo "bypass sigterm"
           else
             # Submit a new job to the queue
             echo "Requeuing " $SLURM_JOB_ID
             scontrol requeue $SLURM_JOB_ID
           fi
        }


        # Install signal handler
        trap 'trap_handler USR1' USR1
        trap 'trap_handler TERM' TERM
    """
