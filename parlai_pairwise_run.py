from parlai_internal.mturk.tasks.pairwise_dialogue_eval.run import main as run_main, make_flags

def set_args():
    args = make_flags()
    args['dialogs_path'] = '/private/home/wangalexc/projects/ParlAI/data/pairwise_eval'
    args['model_comparisons'] = [('lic', 'hf')]
    args['onboarding_tasks'] = [('3WETL7AQWUVO773XHMLZZGBURJE53C', '3II4UPYCOKUBILOSU3FEA0SXC91QDF', 'qual1')]
    args['task_description'] = {'num_subtasks': 5, 'question': args['question']}

    # Main ParlAI Mturk options
    args['num_conversations'] = int(len(args['model_comparisons']) * args['pairs_per_matchup'] / 4)
    args['block_qualification'] = 'testytestytesttest2'
    args['assignment_duration_in_seconds'] = 600
    args['reward'] = 0.5
    args['max_hits_per_worker'] = 1
    args['annotations_per_pair'] = 1


    # Additional args that can be set - here we show the default values.
    # For a full list, refer to run.py & the ParlAI/parlai/params.py
    # args['is_sandbox'] = True
    # args['annotations_per_pair'] = 1
    # args['pairs_per_matchup'] = 160
    # args['seed'] = 42
    # args['s1_choice'] = 'I would prefer to talk to <Speaker 1>'
    # args['s2_choice'] = 'I would prefer to talk to <Speaker 2>'
    # args['question'] = 'Who would you prefer to talk to for a long conversation?'
    # args['block_on_onboarding'] = True

    return args

if __name__ == '__main__':
    args = set_args()
    run_main(args)
