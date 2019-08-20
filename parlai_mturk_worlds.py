#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import time
import random
import _pickle as pkl
import pandas as pd

from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
import parlai.mturk.core.mturk_utils as mturk_utils
from parlai_choice_config import task_config as config


DATA_DIR = "/private/home/wangalexc/projects/qags/data/mturk/pair-judgments"
CHOOSER = 'Chooser'
ONBOARD_MSG = '\nWelcome! \
        This HIT involves reading a news article, \
        followed by two summaries of the article. \
        The task is to select the better overall summary. \
        You will first be presented with a training task. \
        If you perform well enough on the the training, \
        you will proceed to the main task. \
        Type anything and press "Send" to continue.\n'
PICK_BEST_MSG = '\nPlease read the article, and select \
                 the summary you think is <b>TODO</b>.\
                  Then, please explain why you chose that comment \
                  in the chat box below.'
TIMEOUT_MSG = '<b> The other person has timed out. \
        Please click the "Done with this HIT" button below to finish this HIT.\
        </b>'
CHAT_ENDED_MSG = 'You are done with {} articles. Thanks for your time! \nPlease \
        click <span style="color:blue"><b>Done with this HIT</b> </span> \
        button below to finish this HIT.'
WAITING_MSG = 'Please wait...'


def load_csv(data_file):
    """ """
    raw = pd.read_csv(data_file)
    rows = [r[0] for r in raw.iterrows()]
    return rows


class ExampleGenerator(object):
    """Retrieve Example from Personality-Captions Dataset"""

    def __init__(self, opt, data_path=None):
        self.opt = opt
        handle = './examples_stack{}{}{}.pkl'.format(
            '_sandbox' if opt['is_sandbox'] else '',
            opt['compare_key_1'],
            opt['compare_key_2'],
        )
        self.examples_idx_stack_path = os.path.join(os.getcwd(), handle)
        #build_pc(opt)
        if data_path is None:
            data_path = opt.get('eval_data_path')
        with open(data_path) as f:
            self.data = json.load(f)

        if os.path.exists(self.examples_idx_stack_path):
            with open(self.examples_idx_stack_path, 'rb') as handle:
                self.idx_stack = pkl.load(handle)
        else:
            self.idx_stack = []
            self.add_idx_stack()
            self.save_idx_stack()

    def add_idx_stack(self):
        stack = list(range(len(self.data)))
        random.seed()
        random.shuffle(stack)
        self.idx_stack = stack + self.idx_stack

    def pop_example(self):
        if len(self.idx_stack) == 0:
            self.add_idx_stack()
        idx = self.idx_stack.pop()
        ex = self.data[idx]
        return (idx, ex)

    def push_example(self, idx):
        self.idx_stack.append(idx)

    def save_idx_stack(self):
        with open(self.examples_idx_stack_path, 'wb') as handle:
            pkl.dump(self.idx_stack, handle)


class OnboardWorld(MTurkOnboardWorld):
    def parley(self):
        ad = {}
        ad['id'] = 'System'
        ad['text'] = ONBOARD_MSG
        self.mturk_agent.observe(ad)
        self.mturk_agent.act()
        self.episode_done = True


class SoloWorld(MTurkTaskWorld):
#class SoloWorld(MultiAgentDialogWorld):
    """
    World that asks a user 5 math questions, first from a test set if the user
    is entering for the first time, and then randomly for all subsequent times

    Users who don't get enough correct in the test set are assigned a
    qualification that blocks them from completing more HITs during shutdown

    Demos functionality of filtering workers with just one running world.

    Similar results could be achieved by using two worlds where the first acts
    as just a filter and gives either a passing or failing qualification. The
    second would require the passing qualification. The first world could then
    be runnable using the --unique flag.
    """

    def __init__(self, opt, mturk_agent, qualification_id, firstTime):
        self.mturk_agent = mturk_agent
        self.agents = [mturk_agent] # play nice with multiagent code
        self.qualification_id = qualification_id
        self.firstTime = firstTime
        self.test_set = json.load(open(os.path.join(DATA_DIR, "mturk-trg-vs-noisytrg-nex5.json")))
        self.max_resp_time = opt['max_resp_time']  # in secs
        self.episode_done = False
        self.correct = 0
        self.cur_idx = 0
        self.data = []
        self.n_qsts = opt['n_qsts']
        self.ck1 = opt.get('compare_key_1')
        self.ck2 = opt.get('compare_key_2')
        self.opt = opt

    def episode_done(self):
        return self.episode_done

    def parley(self):
        """ """

        control_msg = {'episode_done': False, 'id': 'SYSTEM'}
        agent = self.agents[0]

        while self.cur_idx < self.n_qsts:
            print(f'Task {self.cur_idx}')
            # Send image to turker
            control_msg['description'] = config['task_description']
            if self.firstTime:
                example = self.test_set[self.cur_idx]
                comments = [(ck, example[ck]) for ck in ['trg', 'noise']]
            else:
                self.example_num, example = agent.example_generator.pop_example()
                comments = [(ck, example[ck]) for ck in [self.ck1, self.ck2]]
            control_msg['article'] = example['article'] #self.questions[self.cur_idx][1]

            # Setup comments for ranking
            random.shuffle(comments)
            control_msg['comments'] = [c[1] for c in comments]

            best_pick = None
            control_msg['text'] = PICK_BEST_MSG.format(self.cur_idx + 1)
            control_msg['new_eval'] = True
            agent.observe(validate(control_msg))
            time.sleep(1)

            act = agent.act(timeout=self.max_resp_time)
            # First timeout check
            self.check_timeout(act)
            if self.episode_done:
                break
            try:
                best_idx = int(act['chosen'])
                reason = act['text']
                best_pick = comments[best_idx]
            except Exception:
                # Agent disconnected
                break

            if self.firstTime: # in testing mode
                if (best_idx == 2 and self.questions[self.cur_idx][-1] == 'trg') or \
                   (best_idx == 1 and self.questions[self.cur_idx][-2] == 'trg'):
                    self.correct += 1
            else:
                example['choices'] = comments
                example['best_pick'] = best_pick
                example['reason'] = reason
                self.data.append(example)
                self.cur_idx += 1

        if self.cur_idx == len(self.questions):
            control_msg['text'] = CHAT_ENDED_MSG.format(self.num_images)
            agent.observe(validate(control_msg))
        self.episode_done = True

        return

    def check_timeout(self, act):
        if act['text'] == '[TIMEOUT]' and act['episode_done']:
            control_msg = {'episode_done': True}
            control_msg['id'] = 'SYSTEM'
            control_msg['text'] = TIMEOUT_MSG
            for ag in self.agents:
                if ag.id != act['id']:
                    ag.observe(validate(control_msg))
            self.episode_done = True
            return True
        elif act['text'] == '[DISCONNECT]':
            self.episode_done = True
            return True
        else:
            return False

    def report(self):
        pass

    def save_data(self):
        convo_finished = True
        for ag in self.agents:
            if (
                ag.hit_is_abandoned
                or ag.hit_is_returned
                or ag.disconnected
                or ag.hit_is_expired
            ):
                convo_finished = False
        if not convo_finished:
            ag.example_generator.push_example(self.example_num)
            print("\n**Push image {} back to stack. **\n".format(self.example_num))
        self.agents[0].example_generator.save_idx_stack()
        data_path = self.opt['data_path']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if convo_finished:
            filename = os.path.join(
                data_path,
                '{}_{}_{}.pkl'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type,
                ),
            )
        else:
            filename = os.path.join(
                data_path,
                '{}_{}_{}_incomplete.pkl'.format(
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type,
                ),
            )
        pkl.dump(
            {
                'data': self.data,
                'worker': self.agents[0].worker_id,
                'hit_id': self.agents[0].hit_id,
                'assignment_id': self.agents[0].assignment_id,
            },
            open(filename, 'wb'),
        )
        print('{}: Data successfully saved at {}.'.format(filename))

    def review_work(self):
        pass

        #global review_agent

        #def review_agent(ag):
        #    pass  # auto approve 5 days

        #Parallel(n_jobs=len(self.agents), backend='threading')(
        #    delayed(review_agent)(agent) for agent in self.agents
        #)

    def shutdown(self):
        """Shutdown all mturk agents in parallel, otherwise if one mturk agent
        is disconnected then it could prevent other mturk agents from
        completing.
        """

        """
        Here is where the filtering occurs. If a worker hasn't successfully
        answered all the questions correctly, they are given the qualification
        that marks that they should be blocked from this task.
        """

        if self.firstTime and self.correct != len(self.questions):
            mturk_utils.give_worker_qualification(
                self.mturk_agent.worker_id,
                self.qualification_id,
                is_sandbox=self.opt['is_sandbox'],
            )
        self.mturk_agent.shutdown()

        #global shutdown_agent

        #def shutdown_agent(agent):
        #    agent.shutdown()

        #Parallel(n_jobs=len(self.agents), backend='threading')(
        #    delayed(shutdown_agent)(agent) for agent in self.agents
        #)

