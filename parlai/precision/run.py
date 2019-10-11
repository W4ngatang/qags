""" Run precision mturk task """

from datetime import datetime

from main import main as run_main, make_flags
from config import task_config

def set_args():
    """ """
    args = make_flags()

    curr_time = datetime.now()
    out_file = f"mturk_data.{curr_time.strftime('%m%d%H%M')}.jsonl"
    args['out_file'] = f'/home/awang/projects/qags/data/mturk/summary/precision/{out_file}'
    args['dialogs_path'] = '/home/awang/projects/qags/data/mturk/summary'
    shard_n = 4
    args['model_comparisons'] = [
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

                                 (f'src_para_nex500_ref_order_shard{shard_n}', f'bus_sent_nex500_ref_order_shard{shard_n}'),
                                 #(f'src_para_nex500_ref_order_shard{shard_n}', f'fas_sent_nex500_ref_order_shard{shard_n}'),
                                 #(f'src_para_nex500_ref_order_shard{shard_n}', f'pgc_sent_nex500_ref_order_shard{shard_n}'),
                                 #(f'src_para_nex500_ref_order_shard{shard_n}', f'trg_sent_nex500_ref_order_shard{shard_n}'),

                                ]
    args['pairs_per_matchup'] = 100
    args['annotations_per_pair'] = 1 # 1


    # TODO(Alex): CHANGE ME!!!
    # NOTE(Alex): TURN ON HIT QUALIFICATIONS
    args['is_sandbox'] = False
    args['block_qualification'] = 'aw_block_qags_precision_r8'
    args['min_time_threshold'] = 15

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
    args['num_conversations'] = len(args['model_comparisons']) * args['pairs_per_matchup'] * args['annotations_per_pair']
    args['assignment_duration_in_seconds'] = 600
    args['reward'] = 1.00 # in dollars
    args['max_hits_per_worker'] = 100

    # Additional args that can be set - here we show the default values.
    # For a full list, refer to run.py & the ParlAI/parlai/params.py
    # args['seed'] = 42

    return args

if __name__ == '__main__':
    args = set_args()
    run_main(args, task_config)
