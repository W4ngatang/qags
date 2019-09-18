""" Run precision mturk task """
from qags_main import main as run_main, make_flags
from precision.config import task_config

def set_args():
    """ """
    args = make_flags()
    args['dialogs_path'] = '/home/awang/projects/qags/data/mturk/summary'
    args['out_file'] = '/home/awang/projects/qags/data/mturk/data.jsonl'
    args['model_comparisons'] = [('src_para', 'bus_sent')]
    args['pairs_per_matchup'] = 1
    args['annotations_per_pair'] = 1

    # Task definition
    args['mode'] = 'precision'
    args['question'] = 'Is the sentence factually supported by the article ?'
    args['s1_choice'] = ''
    args['s2_choice'] = ''
    args['task_description'] = {'num_subtasks': 1, 'question': args['question']}

    # Onboarding
    args['onboarding_model_comparison'] = 'onboard_precision_para,onboard_precision_sent'
    args['onboarding_tasks'] = [
        (("onboard-para", 0, -1), ("onboard-sent", 0, 0), 'qual1'),
        (("onboard-para", 0, -1), ("onboard-sent", 0, 1), 'qual1'),
        (("onboard-para", 0, -1), ("onboard-sent", 0, 2), 'qual1'),
        (("onboard-para", 0, -1), ("onboard-sent", 0, 3), 'qual1'),
        ]
    args['comparisons_per_hit'] = 4
    args['block_qualification'] = 'aw_block_qags_precision_r0'
    args['block_on_onboarding'] = True
    args['onboarding_threshold'] = 1.0


    # HIT options
    #args['num_conversations'] = int(len(args['model_comparisons']) * args['pairs_per_matchup'] / 4)
    # add 1 for onboarding task HIT
    args['num_conversations'] = (len(args['model_comparisons']) * args['pairs_per_matchup']) + 1
    args['assignment_duration_in_seconds'] = 600
    args['reward'] = 0.5
    args['max_hits_per_worker'] = 100

    # Additional args that can be set - here we show the default values.
    # For a full list, refer to run.py & the ParlAI/parlai/params.py
    args['is_sandbox'] = True
    # args['seed'] = 42

    return args

if __name__ == '__main__':
    args = set_args()
    run_main(args, task_config)
