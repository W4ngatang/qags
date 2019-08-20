#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import pandas as pd

from parlai.core.worlds import validate
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
import parlai.mturk.core.mturk_utils as mturk_utils



DATA_DIR = "/private/home/wangalexc/projects/qags/data/mturk/pair-judgments"


def load_csv(data_file):
    """ """
    raw = pd.read_csv(data_file)
    rows = [r[0] for r in raw.iterrows()]
    return rows


class QualificationFlowOnboardWorld(MTurkOnboardWorld):
    def parley(self):
        ad = {}
        ad['id'] = 'System'
        # TODO(Alex): update onboarding text
        ad['text'] = (
            'This HIT involves reading a news article, '
            'and then comparing two summaries of that article. '
            'You will first go through a training phase. '
            'If you pass, the next task will be a real one rather than the test one.'
            '\n'
            'Send anything to get started.'
        )
        self.mturk_agent.observe(ad)
        self.mturk_agent.act()
        self.episodeDone = True


class QualificationFlowSoloWorld(MTurkTaskWorld):
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

    test_set = load_csv(os.path.join(DATA_DIR, "mturk-trg-vs-noisytrg-nex5.csv"))
    collector_agent_id = 'System'

    def __init__(self, opt, mturk_agent, qualification_id, firstTime):
        self.mturk_agent = mturk_agent
        self.firstTime = firstTime
        if not firstTime:
            self.questions = self.generate_questions(5)
        else:
            self.questions = self.test_set
        self.episodeDone = False
        self.correct = 0
        self.curr_question = 0
        self.qualification_id = qualification_id
        self.opt = opt

    def generate_questions(self, num):
        questions = load_csv(os.path.join(DATA_DIR, "mturk-bus-vs-fan-nex5.csv"))[:num]
        return questions

    def parley(self):
        if self.curr_question == len(self.questions):
            ad = {
                'episode_done': True,
                'id': self.__class__.collector_agent_id,
                'text': 'Thank you for your answers!',
            }
            self.mturk_agent.observe(validate(ad))
            self.episodeDone = True
        else:
            ad = {
                'episode_done': True,
                'id': self.__class__.collector_agent_id,
                'text': self.questions[self.curr_question][1],
            }
            self.mturk_agent.observe(validate(ad))
            answer = self.mturk_agent.act()
            if self.firstTime:
                if (answer['text'] == 2 and self.questions[self.curr_question][-1] == 'trg') or \
                   (answer['text'] == 1 and self.questions[self.curr_question][-2] == 'trg'):
                    self.correct += 1
            else:
                if answer['text'] == 1: #self.questions[self.curr_question][1]:
                    self.correct += 1
            self.curr_question += 1

    def episode_done(self):
        return self.episodeDone

    def report(self):
        pass

    def shutdown(self):
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

    def review_work(self):
        pass
