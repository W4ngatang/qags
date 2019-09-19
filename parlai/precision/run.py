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
    args['model_comparisons'] = [('src_para', 'bus_sent'),
                                 #('src_para', 'fas_sent'),
                                 #('src_para', 'pgc_sent')
                                 #('src_para', 'trg_sent')
                                ]
    args['pairs_per_matchup'] = 10 # 100
    args['annotations_per_pair'] = 1 # 1


    # TODO(Alex): CHANGE ME!!!
    args['is_sandbox'] = False

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
        (("onboard-pcs-para", 0, -1), ("onboard-pcs-sent", 0, 0), 'qual-pcs'),
        (("onboard-pcs-para", 0, -1), ("onboard-pcs-sent", 0, 1), 'qual-pcs'),
        (("onboard-pcs-para", 0, -1), ("onboard-pcs-sent", 0, 2), 'qual-pcs'),
        (("onboard-pcs-para", 0, -1), ("onboard-pcs-sent", 0, 3), 'qual-pcs'),
        ]
    args['comparisons_per_hit'] = 4
    args['block_qualification'] = 'aw_block_qags_precision_r6'
    args['block_on_onboarding'] = True
    args['onboarding_threshold'] = 1.0

    # HIT options
    # add 1 for onboarding task HIT
    args['num_conversations'] = (len(args['model_comparisons']) * args['pairs_per_matchup']) + 1
    args['assignment_duration_in_seconds'] = 600
    args['reward'] = 2.00 # in dollars
    args['max_hits_per_worker'] = 100

    # Additional args that can be set - here we show the default values.
    # For a full list, refer to run.py & the ParlAI/parlai/params.py
    # args['seed'] = 42

    return args

if __name__ == '__main__':
    args = set_args()
    run_main(args, task_config)
