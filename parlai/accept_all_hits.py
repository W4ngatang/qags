""" """

import os
import ipdb
from datetime import datetime

from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_utils import setup_aws_credentials
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler

PATHS = {
         'sandbox': ('/home/awang/projects/ParlAI/parlai/mturk/run_data/pmt_sbdata.db',
                     '/home/awang/projects/ParlAI/parlai/mturk/run_data/sandbox/'),
         'live': ('/home/awang/projects/ParlAI/parlai/mturk/run_data/pmt_data.db',
                     '/home/awang/projects/ParlAI/parlai/mturk/run_data/live/'),
        }


def main(opt):
    setup_aws_credentials()
    if opt['no_sandbox']:
        db_file, run_dir = PATHS['live']
    else:
        db_file, run_dir = PATHS['sandbox']
    assert os.path.exists(db_file), f"DB file {db_file} doesn't exist!"
    assert os.path.isdir(run_dir), f"run directory {run_dir} doesn't exist!"
    db = MTurkDataHandler(file_name=db_file)
    mturk_manager = MTurkManager(opt, [])

    # Get run IDs
    if opt['run_ids'] is None:
        run_ids = list(os.listdir(run_dir))
    else:
        run_ids = opt['run_ids'].split(',')

    def get_run_id_data(run_ids, db):
        """ """
        print(f"Found following run IDs: ")
        n_hits = 0
        for run_id in run_ids:
            run_data = db.get_run_data(run_id)
            start_time = datetime.fromtimestamp(run_data['launch_time'])
            hits = db.get_pairings_for_run(run_id)
            n_hits += len(hits)
            print(f"{run_id}: {len(hits)} HITS, started {start_time}")
        print(f"Total {n_hits} HITS over {len(run_ids)} runs")


    def approve_run_hits(run_id, db, manager):
        """ """
        to_approve = []
        n_approved = 0
        hits = db.get_pairings_for_run(run_id)
        for hit in hits:
            if hit['conversation_id'] is None:
                continue
            try:
                full_data = db.get_full_conversation_data(run_id, hit['conversation_id'], False)
            except FileNotFoundError:
                continue

            data = next(iter(full_data['worker_data'].values()))
            try:
                n_approved += 1
                to_approve.append(data['assignment_id'])
                print(f"Approved {data['assignment_id']}")
            except:
                print(f"Failed to approve {data['assignment_id']}")
        print(f"Run ID {run_id}: to approve {n_approved} HITs")
        conf = input("Confirm? (y/n): ")
        if conf == "y":
            for asgn_id in to_approve:
                #mturk_manager.approve_work(data['assignment_id'], override_rejection=True)
                pass
            print(f"\tApproved {n_approved} HITs")
        else:
            print("\tCancelled approvals")


    def approve_assignments(asgn_ids, mturk_manager):
        """ """
        for asgn_id in asgn_ids:
            #mturk_manager.approve_work(data['assignment_id'], override_rejection=True)
            pass
        print(f"\tApproved {len(asgn_ids)} HITs")


    def inspect_assignment(asgn_id):
        """ """
        asgn_data = db.get_assignment_data(asgn_id)
        ipdb.set_trace()


    # main loop
    while True:
        get_run_id_data(run_ids, db=db)
        #run_id = input("Enter run ID: ")
        cmd = input("Enter command: ")
        if len(cmd) == 0 or cmd == "exit":
            break
        cmd_parts = cmd.split()
        if cmd_parts[0] == "inspect":
            assert len(cmd_parts), "No assignment ID provided."
            inspect_assignment(cmd_parts[1])
        elif cmd_parts[0] == "approve":
            assert len(cmd_parts), "No run ID provided."
            if run_id in run_ids:
                approve_run_hits(run_id, db=db, manager=mturk_manager)
            else:
                print(f"Run ID {run_id} not found!")

    #for run_id in run_ids:
    #    approve_run_hits(run_id)



if __name__ == '__main__':
    parser = ParlaiParser(False, False)
    parser.add_mturk_args()
    parser.add_argument('--run_ids', type=str, default=None, help='comma separated run ids')
    parser.add_argument('--no_sandbox', action='store_true', help='comma separated run ids')
    opt = parser.parse_args()
    main(opt)

# python parlai_internal/mturk/tasks/pairwise_dialogue_eval/scripts/accept_all_hits.py --run_ids pairwise_dialogue_eval_1556568703,pairwise_dialogue_eval_1556853821
