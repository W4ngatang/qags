#from parlai_internal.mturk.tasks.pairwise_dialogue_eval.run import main as run_main, make_flags
from qags_main import main as run_main, make_flags
from qags_recall_config import task_config

def set_args():
    args = make_flags()
    #args['dialogs_path'] = '/private/home/wangalexc/projects/qags/data/mturk/pairwise_eval'
    #args['model_comparisons'] = [('dummy-sents', 'dummy-passages')]
    args['dialogs_path'] = '/private/home/wangalexc/projects/qags/data/mturk/summary'
    args['model_comparisons'] = [('bus_para', 'src_sent')]
    #args['pair_data'] = [('src_para', 'bus_sents')]
    args['pairs_per_matchup'] = 10
    args['annotations_per_pair'] = 1

    # Task definition
    args['mode'] = 'recall'
    args['question'] = 'Is the main idea of the sentence captured by the article ?'
    args['s1_choice'] = ''
    args['s2_choice'] = ''
    args['task_description'] = {'num_subtasks': 1, 'question': args['question']}

    # Onboarding
    args['onboarding_model_comparison'] = 'greedy_model,human_eval'
    args['onboarding_tasks'] = [('3WETL7AQWUVO773XHMLZZGBURJE53C', '3II4UPYCOKUBILOSU3FEA0SXC91QDF', 'qual1')]
    args['block_qualification'] = None #'testytestytesttest2'
    args['block_on_onboarding'] = False


    # HIT options
    args['num_conversations'] = int(len(args['model_comparisons']) * args['pairs_per_matchup'] / 4)
    args['assignment_duration_in_seconds'] = 600
    args['reward'] = 0.5
    #args['comparisons_per_hit'] = 3
    args['max_hits_per_worker'] = 5

    # Additional args that can be set - here we show the default values.
    # For a full list, refer to run.py & the ParlAI/parlai/params.py
    args['is_sandbox'] = True
    # args['seed'] = 42

    return args

if __name__ == '__main__':
    args = set_args()
    run_main(args, task_config)
