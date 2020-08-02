#!/usr/bin/env python

from functools import reduce
import sys
import sweep
from sweep import hyperparam


def get_lr_str(val):
    uniq_lr = reduce(
        lambda x, y: x + y if x[-1] != y[-1] else x,
        map(lambda x: [x], val.split(','))
    )
    return ','.join(uniq_lr)


def get_filter_str(val):
    f = eval(val)
    s = f'cf{f[0][1]}'
    return s

max_update=30500
def get_grid(args):
    return [
        hyperparam('--save-interval', 1),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--warmup', 0.1),
        hyperparam('--arch', 'finetuning_squad', save_dir_key=lambda val: val),
        hyperparam('--task', 'squad'),
        hyperparam('--max-update', [
            max_update
        ], save_dir_key=lambda val: f'mxup{val}'),
        hyperparam('--data-file' ,'/private/home/yinhanliu/data/squad_bert/dev-v1.1.json'),
        hyperparam('--optimizer', 'bert_adam', save_dir_key=lambda val: val),
        hyperparam('--lr', [
          3e-05, 2e-05
        ], save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--t-total', max_update),
        hyperparam('--bert-path','/checkpoint/yinhanliu/20190322/hf_bert_implement/bert512.eps-06_0.0002/checkpoint_best.pt',
            save_dir_key=lambda val: f'bert'),

        hyperparam('--criterion', ['squad'], save_dir_key=lambda val: f'crs_ent'),
        hyperparam('--max-tokens', [
            1334,
        ], save_dir_key=lambda val: f'mxtk{val}'),
        hyperparam('--seed', [3,4,5,6], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--skip-invalid-size-inputs-valid-test'),
        hyperparam('--log-format', 'json'),
        hyperparam('--log-interval', [500]),
        hyperparam('--min-lr', 1e-09),
        hyperparam('--model-dim', 768),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    # if config['--seq-beam'].current_value <= 8:
    #    config['--max-tokens'].current_value = 400
    # else:
    #    config['--max-tokens'].current_value = 300


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
