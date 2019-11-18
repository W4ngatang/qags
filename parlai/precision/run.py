""" Run precision mturk task """

import math
from datetime import datetime
import ipdb

from main import main as run_main, make_flags
from config import task_config

def set_args():
    """ """
    args = make_flags()

    curr_time = datetime.now()
    out_dir = f"/home/awang/projects/qags/data/mturk/summary/precision"
    out_file = f"{curr_time.strftime('%m%d%H%M')}.mturk_data.jsonl"
    bonus_file = f"{curr_time.strftime('%m%d%H%M')}.bonuses.csv"
    args['out_file'] = f'{out_dir}/{out_file}'
    args['bad_worker_file'] = f'/home/awang/projects/qags/data/mturk/bad_workers.txt'
    args['bonus_file'] = f'{out_dir}/{bonus_file}'

    args['dialogs_path'] = '/home/awang/projects/qags/data/mturk/summary'
    shard_n = 5
    args['model_comparisons'] = [
                                 #(f'src_para_nex5_randorder_shard{shard_n}', f'bart_sent_nex5_randorder_shard{shard_n}'),
                                 #(f'src_para_nex5_randorder_shard{shard_n}', f'bus_sent_nex5_randorder_shard{shard_n}'),

                                 #('src_para_short', 'bus_sent_short'),
                                 #('src_para_short', 'fas_sent_short'),
                                 #('src_para_short', 'pgc_sent_short'),
                                 #('src_para_short', 'trg_sent_short')

                                 #('src_para_short_attn', 'bus_sent_short_attn'),
                                 #('src_para_short_attn', 'fas_sent_short_attn'),
                                 #('src_para_short_attn', 'pgc_sent_short_attn'),
                                 #('src_para_short_attn', 'trg_sent_short_attn'),

                                 #('src_para_short_attn_v2', 'bus_sent_short_attn_v2'),
                                 #('src_para_short_attn_v2', 'fas_sent_short_attn_v2'),
                                 #('src_para_short_attn_v2', 'pgc_sent_short_attn_v2'),
                                 #('src_para_short_attn_v2', 'trg_sent_short_attn_v2'),

                                 #(f'src_para_nex500_ref_order_shard{shard_n}', f'bus_sent_nex500_ref_order_shard{shard_n}'),
                                 #(f'src_para_nex500_ref_order_shard{shard_n}', f'fas_sent_nex500_ref_order_shard{shard_n}'),
                                 #(f'src_para_nex500_ref_order_shard{shard_n}', f'pgc_sent_nex500_ref_order_shard{shard_n}'),
                                 #(f'src_para_nex500_ref_order_shard{shard_n}', f'trg_sent_nex500_ref_order_shard{shard_n}'),

                                 (f'src_para_nex1000_randorder_shard{shard_n}', f'bus_sent_nex1000_randorder_shard{shard_n}'),
                                 #(f'src_para_nex1000_randorder_shard{shard_n}', f'fas_sent_nex1000_randorder_shard{shard_n}'),
                                 #(f'src_para_nex1000_randorder_shard{shard_n}', f'pgc_sent_nex1000_randorder_shard{shard_n}'),
                                 #(f'src_para_nex1000_randorder_shard{shard_n}', f'trg_sent_nex1000_randorder_shard{shard_n}'),

                                 #(f'src_para_nex100_randorder_shard{shard_n}', f'bus_sent_nex100_randorder_shard{shard_n}'),
                                 #(f'src_para_nex100_randorder_shard{shard_n}', f'fas_sent_nex100_randorder_shard{shard_n}'),
                                 #(f'src_para_nex100_randorder_shard{shard_n}', f'pgc_sent_nex100_randorder_shard{shard_n}'),
                                 #(f'src_para_nex100_randorder_shard{shard_n}', f'trg_sent_nex100_randorder_shard{shard_n}'),

                                ]
    args['pairs_per_matchup'] = 50
    args['annotations_per_pair'] = 3

    # TODO(Alex): CHANGE ME!!!
    args['is_sandbox'] = False
    args['qual_percent_hits_approved'] = 98
    args['qual_n_hits_approved'] = 1000
    args['min_time_threshold'] = 30
    args['block_qualification'] = 'aw_block_qags_precision_r8'


    # Task definition
    args['mode'] = 'precision'
    args['question'] = 'Is the sentence factually supported by the article ?'
    args['s1_choice'] = ''
    args['s2_choice'] = ''
    args['task_description'] = {
                                'num_subtasks': 1,
                                'question': args['question']
                               }

    # Onboarding
    args['onboarding_model_comparison'] = 'onboard_precision_para,onboard_precision_sent'
    args['onboarding_tasks'] = [
        #(("onboard-pcs-para", 1, -1), ("onboard-pcs-sent", 1, 0), 'qual-pcs'),
        #(("onboard-pcs-para", 1, -1), ("onboard-pcs-sent", 1, 1), 'qual-pcs'),
        #(("onboard-pcs-para", 1, -1), ("onboard-pcs-sent", 1, 2), 'qual-pcs'),
        #(("onboard-pcs-para", 0, -1), ("onboard-pcs-sent", 0, 3), 'qual-pcs'),
        ]
    args['comparisons_per_hit'] = 1 #len(args['onboarding_tasks'])
    args['block_on_onboarding'] = True
    args['onboarding_threshold'] = 1.0

    # HIT options
    # add 1 for onboarding task HIT
    # Manager workers by creating HITS until num_conversations is reached, including
    # onboarding tasks
    #args['num_conversations'] = 10 + (len(args['model_comparisons']) * args['pairs_per_matchup'] * args['annotations_per_pair'])
    args['num_conversations'] = math.ceil((len(args['model_comparisons']) * args['pairs_per_matchup'] * args['annotations_per_pair']) * 1.33)
    args['assignment_duration_in_seconds'] = 1800
    args['reward'] = 0.15 # in dollars
    args['bonus_reward'] = 0.85 # in dollars
    args['max_hits_per_worker'] = 100

    # Additional args that can be set - here we show the default values.
    # For a full list, refer to run.py & the ParlAI/parlai/params.py
    # args['seed'] = 42
    args['verbose'] = False
    args['is_debug'] = False

    return args

if __name__ == '__main__':
    args = set_args()
    run_main(args, task_config)
