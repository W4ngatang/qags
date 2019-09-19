""" Run recall mturk task """
from datetime import datetime

from main import main as run_main, make_flags
from config import task_config

def set_args():
    args = make_flags()
    curr_time = datetime.now()
    out_file = f"mturk_data.{curr_time.strftime('%m%d%H%M')}.jsonl"
    args['out_file'] = f'/home/awang/projects/qags/data/mturk/summary/recall/{out_file}'
    args['dialogs_path'] = '/home/awang/projects/qags/data/mturk/summary'
    args['model_comparisons'] = [
            ('bus_para', 'src_sent'),
            #('fas_para', 'src_sent'),
            #('pgc_para', 'src_sent'),
            #('trg_para', 'src_sent'),
        ]
    args['pairs_per_matchup'] = 1 # 100
    args['annotations_per_pair'] = 1

    # Task definition
    args['mode'] = 'recall'
    args['question'] = 'Is the main idea of the sentence captured by the article ?'
    args['s1_choice'] = ''
    args['s2_choice'] = ''
    args['task_description'] = {'num_subtasks': 1, 'question': args['question']}

    # Onboarding
    args['onboarding_model_comparison'] = 'onboard_recall_para,onboard_recall_sent'
    args['onboarding_tasks'] = [
        (("onboard-rcl-para", 0, -1), ("onboard-rcl-sent", 0, 0), 'qual-rcl'),
        (("onboard-rcl-para", 0, -1), ("onboard-rcl-sent", 0, 1), 'qual-rcl'),
        (("onboard-rcl-para", 0, -1), ("onboard-rcl-sent", 0, 2), 'qual-rcl'),
        (("onboard-rcl-para", 0, -1), ("onboard-rcl-sent", 0, 3), 'qual-rcl'),
        ]
    args['comparisons_per_hit'] = 4
    args['block_qualification'] = 'aw_block_qags_recall_r6'
    args['block_on_onboarding'] = True
    args['onboarding_threshold'] = 1.0

    # HIT options
    args['num_conversations'] = (len(args['model_comparisons']) * args['pairs_per_matchup']) + 1
    args['assignment_duration_in_seconds'] = 600
    args['reward'] = 2.00
    args['max_hits_per_worker'] = 100

    # Additional args that can be set - here we show the default values.
    # For a full list, refer to run.py & the ParlAI/parlai/params.py
    args['is_sandbox'] = True
    # args['seed'] = 42

    return args

if __name__ == '__main__':
    args = set_args()
    run_main(args, task_config)
