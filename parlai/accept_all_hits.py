""" """

import os
import time
import glob
import json
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import ipdb

from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_utils import setup_aws_credentials
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.mturk_utils as mturk_utils

PATHS = {
         'sandbox': ('/home/awang/projects/ParlAI/parlai/mturk/run_data/pmt_sbdata.db',
                     '/home/awang/projects/ParlAI/parlai/mturk/run_data/sandbox/'),
         'live': ('/home/awang/projects/ParlAI/parlai/mturk/run_data/pmt_data.db',
                     '/home/awang/projects/ParlAI/parlai/mturk/run_data/live/'),
        }

BONUS_MSG = 'Bonus for performing HIT well!'
BAD_RESPONSES = ['[RETURNED]', '[DISCONNECT]']
CHOICE2ANS = {'2': 'no', '1': 'yes'}

def main(opt):
    setup_aws_credentials()
    if opt['no_sandbox']:
        db_file, all_runs_dir = PATHS['live']
        opt['is_sandbox'] = False
    else:
        db_file, all_runs_dir = PATHS['sandbox']
    assert os.path.exists(db_file), f"DB file {db_file} doesn't exist!"
    assert os.path.isdir(all_runs_dir), f"run directory {all_runs_dir} doesn't exist!"
    db = MTurkDataHandler(file_name=db_file)
    mturk_manager = MTurkManager(opt, [])
    client = mturk_utils.get_mturk_client(not opt['no_sandbox'])


    # Get run IDs
    if opt['run_ids'] is None:
        run_ids = list(os.listdir(all_runs_dir))
        run2worker = defaultdict(lambda: dict())
        worker2run = defaultdict(lambda: dict())
        for run_id in run_ids:
            run_dir = os.path.join(all_runs_dir, run_id)
            hits = os.listdir(run_dir)
            for hit in hits:
                # t_*/workers/{WORKER_ID}.json
                resps = os.listdir(f"{run_dir}/{hit}/workers/")
                assert len(resps) == 1, "More than one response found!"
                worker_id = resps[0].split('.')[0]
                worker_data = json.load(open(os.path.join(run_dir, hit, "workers", resps[0])))
                run2worker[run_id][worker_id] = worker_data
                worker2run[worker_id][run_id] = worker_data

    else:
        run_ids = opt['run_ids'].split(',')

    def get_all_hits():
        """ """
        all_hits = []
        resp = client.list_hits()
        all_hits.append(resp['HITs'])
        while 'NextToken' in resp and resp['NextToken']:
            resp = client.list_hits(NextToken=resp['NextToken'])
            all_hits += resp['HITs']
            time.sleep(0.5)
        return all_hits

    def get_run_id_data(run_ids):
        """ """
        print(f"Found following run IDs: ")
        n_hits = 0
        run_data = list()
        for run_id in run_ids:
            run_datum = db.get_run_data(run_id)
            run_data.append((run_id, run_datum))
        run_data.sort(key=lambda x: x[1]['launch_time'])
        for run_id, run_datum in run_data:
            start_time = datetime.fromtimestamp(run_datum['launch_time'])
            hits = db.get_pairings_for_run(run_id)
            n_hits += len(hits)
            print(f"{run_id} {len(hits)} HITS, started {start_time}")
        print(f"Total {n_hits} HITS over {len(run_ids)} runs")


    def approve_run_hits(run_id):
        """ """
        to_approve = []
        n_to_approve, n_approved = 0, 0
        hits = db.get_pairings_for_run(run_id)
        data = []
        for hit in hits:
            if hit['conversation_id'] is None:
                continue
            try:
                full_data = db.get_full_conversation_data(run_id, hit['conversation_id'], False)
            except FileNotFoundError:
                continue

            datum = next(iter(full_data['worker_data'].values()))
            if datum['response']['text'] in BAD_RESPONSES:
                continue
            n_to_approve += 1
            to_approve.append(datum['assignment_id'])
            data.append(datum)
            print(f"To approve: {datum['assignment_id']}")

        print(f"Run ID {run_id}: to approve {n_to_approve} HITs")
        conf = input("Confirm? (y/n): ")
        if conf == "y":
            didnt_approve = list()
            for asgn_id in to_approve:
                try:
                    mturk_manager.approve_work(asgn_id)
                    n_approved += 1
                    print(f"Approved {asgn_id}")
                except:
                    didnt_approve.append(asgn_id)
                    print(f"Failed to approve: {asgn_id}")
            print(f"\tApproved {n_approved} HITs")
            if didnt_approve:
                print(f"\tFailed to approve assignments {','.join(didnt_approve)}")
        else:
            print("\tCancelled approvals")

    def approve_assignment(asgn_id):
        """ """
        conf = input(f"Confirm approving assignment {asgn_id}? (y/n): ")
        if conf == "y":
            try:
                mturk_manager.approve_work(asgn_id, override_rejection=True)
                print(f"\tSuccessfully approved!")
            except:
                print(f"\tFailed to approve.")

        else:
            print("\tCancelled approvals.")

    def award_from_file(bonus_file, msg):
        awards = [r.split(',') for r in open(bonus_file, encoding="utf-8")]
        total_bonus = sum(float(award[-1]) for award in awards)
        conf = input(f"Confirm awarding total bonus ${total_bonus} to {len(awards)} workers? ")
        if conf == "y":
            n_awarded = 0
            amt_awarded = 0.0
            didnt_award = list()
            for award in tqdm.tqdm(awards):
                worker_id, asgn_id, request_tok, bonus_amt = award
                bonus_amt = float(bonus_amt)
                try:
                    mturk_manager.pay_bonus(worker_id=worker_id,
                                            bonus_amount=bonus_amt,
                                            assignment_id=asgn_id,
                                            reason=msg,
                                            unique_request_token=request_tok)
                    n_awarded += 1
                    amt_awarded += bonus_amt
                except:
                    didnt_award.append((worker_id, asgn_id, request_tok, bonus_amt))
                    #print(f"\tFailed to award bonus to {worker_id}")
            print(f"Awarded {amt_awarded} to {n_awarded} workers.")
            if didnt_award:
                for worker_id, asgn_id, request_tok, bonus_amt in didnt_award:
                    print(f"\tFailed to award bonus {bonus_amt} to {worker_id} for assignment {asgn_id} (tok: {request_tok})")
        else:
            print("\tCancelled bonus.")

    def award_bonus(worker_id, bonus_amt, asgn_id, msg, request_tok):
        conf = input(f"Confirm awarding ${bonus_amt} to {worker_id}?")
        if conf == "y":
            try:
                mturk_manager.pay_bonus(worker_id=worker_id,
                                        bonus_amount=bonus_amt,
                                        assignment_id=asgn_id,
                                        reason=msg,
                                        unique_request_token=request_tok)
                print(f"\tSuccessfully approved!")
            except:
                print(f"\tFailed to approve.")
        else:
            print("\tCancelled bonus.")



    def inspect_assignment(asgn_id):
        """ """
        raise NotImplementedError
        #asgn_data = db.get_assignment_data(asgn_id)
        #if asgn_data is None:
        #    print("Assignment ID {asgn_id} not found.")

    def inspect_hit(hit_id):
        """ """
        raise NotImplementedError
        #hit_data = db.get_hit_data(hit_id)
        #if hit_data is None:
        #    print("HIT ID {hit_id} not found.")


    def inspect_run_worker_pair(run_id, worker_id):
        """ """
        worker_data = run2worker[run_id][worker_id]
        asgn_id = worker_data['assignment_id']
        answers = list()
        qsts = list()
        ctx = worker_data['task_data'][0]['conversations'][0]['dialog'][0]['text']
        for task_datum in worker_data['task_data']:
            qst_d = task_datum['conversations'][1]
            qsts.append(qst_d['dialog'][0]['text'])
            if 'answer' in qst_d and 'answer' is not None:
                answers.append(qst_d['answer'])
            else:
                answers.append(None)

        try:
            choices = [CHOICE2ANS[r['speakerChoice']] for r in worker_data['response']['task_data']]
            reasons = [r['textReason'] for r in worker_data['response']['task_data']]
        except KeyError as e:
            print("Key error!")
            print("task_data not in worker response!")
            ipdb.set_trace()

        try:
            pair = db.get_worker_assignment_pairing(worker_id, asgn_id)
            hit_time = pair['task_end'] - pair['task_start']
        except:
            ipdb.set_trace()

        print(f"\nAssignment ID: {worker_data['assignment_id']}")
        print(f"CONTEXT: {ctx}\n")
        for qst, ans, choice, reason in zip(qsts, answers, choices, reasons):
            print(f"QUESTION: {qst}")
            print(f"ANSWER: {ans}")
            print(f"CHOICE: {choice}")
            print(f"REASON: {reason}")
            print()
        print(f"HIT time: {hit_time}")
        resp = input("Accept (y/n) ? ")
        if resp == "y":
            #try:
            #    mturk_manager.approve_work(asgn_id, override_rejection=True)
            #    print("\tApproved!")
            #except:
            #    ipdb.set_trace()
            mturk_manager.approve_work(asgn_id, override_rejection=True)
            print("\tApproved!")

    def inspect_hit_worker_pair(hit_id, worker_id):
        """ """
        resp = client.list_assignments_for_hit(HITId=hit_id)
        all_asgns = list(resp['Assignments'])
        while 'NextToken' in resp and resp['NextToken']:
            resp = client.list_assignments_for_hit(HITId=hit_id,
                                                   NextToken=resp['NextToken'])
            if resp['Assignments']:
                all_asgns.append(resp['Assignments'])
            time.sleep(0.5)

        assert len(all_asgns) == 1, ipdb.set_trace()
        asgn_ids = [a['AssignmentId'] for a in all_asgns]
        run_ids = list()
        worker_runs = worker2run[worker_id]
        for asgn_id in asgn_ids:
            for run_id, run_d in worker_runs.items():
                if run_d['assignment_id'] == asgn_id:
                    run_ids.append(run_id)
        print(f"Assignment ID: {asgn_ids[0]}")
        print(f"Submit date: {all_asgns[0]['SubmitTime'].strftime('%m/%d')}")
        #assert len(run_ids) == 1, ipdb.set_trace()
        #run_id = run_ids[0]
        #asgn_id = asgn_ids[0]
        #worker_data = run2worker[run_id][worker_id]
        #answers = list()
        #qsts = list()
        #ctx = worker_data['task_data'][0]['conversations'][0]['dialog'][0]['text']
        #for task_datum in worker_data['task_data']:
        #    qst_d = task_datum['conversations'][1]
        #    qsts.append(qst_d['dialog'][0]['text'])
        #    if 'answer' in qst_d and 'answer' is not None:
        #        answers.append(qst_d['answer'])
        #    else:
        #        answers.append(None)

        #try:
        #    choices = [CHOICE2ANS[r['speakerChoice']] for r in worker_data['response']['task_data']]
        #    reasons = [r['textReason'] for r in worker_data['response']['task_data']]
        #except KeyError as e:
        #    print("Key error!")
        #    print("task_data not in worker response!")
        #    ipdb.set_trace()

        #try:
        #    pair = db.get_worker_assignment_pairing(worker_id, asgn_id)
        #    hit_time = pair['task_end'] - pair['task_start']
        #except:
        #    ipdb.set_trace()

        #print(f"\nAssignment ID: {worker_data['assignment_id']}")
        #print(f"CONTEXT: {ctx}\n")
        #for qst, ans, choice, reason in zip(qsts, answers, choices, reasons):
        #    print(f"QUESTION: {qst}")
        #    print(f"ANSWER: {ans}")
        #    print(f"CHOICE: {choice}")
        #    print(f"REASON: {reason}")
        #    print()
        #print(f"HIT time: {hit_time}")
        #resp = input("Accept (y/n) ? ")
        #if resp == "y":
        #    try:
        #        mturk_manager.approve_work(asgn_id, override_rejection=True)
        #        print("\tApproved!")
        #    except:
        #        ipdb.set_trace()

    # main loop
    while True:
        print("Enter 'p' to print runs")
        cmd = input("Enter command: ")
        if len(cmd) == 0 or cmd == "exit":
            break
        cmd_parts = cmd.split()
        if cmd_parts[0] == "p":
            get_run_id_data(run_ids)
        elif cmd_parts[0] == "inspect":
            assert len(cmd_parts) == 3, "Insufficient arguments!"
            inspect_run_worker_pair(cmd_parts[1], cmd_parts[2])
        elif cmd_parts[0] in ["get-asgn", 'ga']:
            assert len(cmd_parts) == 3, "Insufficient arguments! Please provide worker_id and ..."
            inspect_hit_worker_pair(cmd_parts[1], cmd_parts[2])
        elif cmd_parts[0] == "inspect-asgn":
            assert len(cmd_parts) > 1, "No assignment ID provided."
            inspect_assignment(cmd_parts[1])
        elif cmd_parts[0] == "inspect-hit":
            assert len(cmd_parts) > 1, "No HIT ID provided."
            inspect_hit(cmd_parts[1])
        elif cmd_parts[0] == "approve":
            assert len(cmd_parts) > 1, "No run ID provided."
            run_id = cmd_parts[1]
            if run_id in run_ids:
                approve_run_hits(run_id)
            else:
                print(f"Run ID {run_id} not found!")
        elif cmd_parts[0] == "approve-asgn":
            assert len(cmd_parts) > 1, "No assignment ID provided."
            approve_assignment(cmd_parts[1])
        elif cmd_parts[0] == "award-from-file":
            assert len(cmd_parts) > 1, "No file provided."
            if not os.path.exists(cmd_parts[1]):
                print(f"File {cmd_parts[1]} not found!")
                continue
            award_from_file(cmd_parts[1], BONUS_MSG)
        elif cmd_parts[0] in ["d", "debug"]:
            ipdb.set_trace()
        else:
            print(f"Command `{cmd}` not understood.")


if __name__ == '__main__':
    parser = ParlaiParser(False, False)
    parser.add_mturk_args()
    parser.add_argument('--run_ids', type=str, default=None, help='comma separated run ids')
    parser.add_argument('--no_sandbox', action='store_true', help='If given, run against live data')
    opt = parser.parse_args()
    main(opt)

# python parlai_internal/mturk/tasks/pairwise_dialogue_eval/scripts/accept_all_hits.py --run_ids pairwise_dialogue_eval_1556568703,pairwise_dialogue_eval_1556853821
