#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
#from parlai.mturk.tasks.qualification_flow_example.worlds import (
#    QualificationFlowOnboardWorld,
#    QualificationFlowSoloWorld,
#)
from parlai_mturk_worlds import (
    OnboardWorld, SoloWorld, ExampleGenerator,
)
from parlai.mturk.core.mturk_manager import MTurkManager
import parlai.mturk.core.mturk_utils as mturk_utils
from parlai.mturk.tasks.qualification_flow_example.task_config import task_config

import os
import json
import random


def main():
    completed_workers = []
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument(
        '-mx_rsp_time',
        '--max_resp_time',
        default=1800,
        type=int,
        help='time limit for entering a dialog message',
    )
    argparser.add_argument(
        '-mx_onb_time',
        '--max_onboard_time',
        type=int,
        default=300,
        help='time limit for turker' 'in onboarding',
    )
    argparser.add_argument(
        '-nq',
        '--n_qsts',
        type=int,
        default=5,
        help='number of images to show \
                           to turker',
    )
    argparser.add_argument(
        '--data-path', type=str, default='/private/home/wangalexc/projects/qags/data/mturk', help='where to save data'
    )
    argparser.add_argument(
        '--eval-data-path',
        type=str,
        default='/private/home/wangalexc/projects/qags/data/mturk/pair-judgments/mturk-bus-vs-fan-nex50.json',
        help='where to load data to rank from. Leave '
        'blank to use Personality-Captions data',
    )
    argparser.add_argument(
        '-ck1',
        '--compare-key-1',
        type=str,
        default='bus',
        help='key of first option to compare',
    )
    argparser.add_argument(
        '-ck2',
        '--compare-key-2',
        type=str,
        default='fan',
        help='key of second option to compare',
    )

    opt = argparser.parse_args()
    directory_path = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(directory_path)
    opt.update(task_config)

    mturk_agent_id = 'Worker'
    mturk_manager = MTurkManager(opt=opt, mturk_agent_ids=[mturk_agent_id])
    example_generator = ExampleGenerator(opt)
    #mturk_manager.setup_server()
    mturk_manager.setup_server(task_directory_path=directory_path)

    qual_name = 'ParlAIExcludeQual{}t{}'.format(
        random.randint(10000, 99999), random.randint(10000, 99999)
    )
    qual_desc = (
        'Qualification for a worker not correctly completing the '
        'first iteration of a task. Used to filter to different task pools.'
    )
    qualification_id = mturk_utils.find_or_create_qualification(
        qual_name, qual_desc, opt['is_sandbox']
    )
    print('Created qualification: ', qualification_id)

    def run_onboard(worker):
        worker.example_generator = example_generator
        world = OnboardWorld(opt, worker)
        while not world.episode_done():
            world.parley()
        world.shutdown()

    mturk_manager.set_onboard_function(onboard_function=run_onboard)

    try:
        mturk_manager.start_new_run()
        agent_qualifications = [
            {
                'QualificationTypeId': qualification_id,
                'Comparator': 'DoesNotExist',
                'RequiredToPreview': True,
            }
        ]

        # Set up the scokets and threads to recieve workers
        mturk_manager.ready_to_accept_workers()

        # Create the hits as specified by command line arguments
        mturk_manager.create_hits(qualifications=agent_qualifications)

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(worker):
            worker[0].id = mturk_agent_id

        global run_conversation

        def run_conversation(mturk_manager, opt, workers):
            mturk_agent = workers[0]
            world = SoloWorld(
                opt=opt,
                mturk_agent=mturk_agent,
                qualification_id=qualification_id,
                firstTime=(mturk_agent.worker_id not in completed_workers),
            )
            while not world.episode_done():
                world.parley()
            world.save_data()
            completed_workers.append(mturk_agent.worker_id)
            world.shutdown()
            world.review_work()

        mturk_manager.start_task(
            eligibility_function=check_worker_eligibility,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation,
        )
    except BaseException:
        raise
    finally:
        mturk_utils.delete_qualification(qualification_id, opt['is_sandbox'])
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()
