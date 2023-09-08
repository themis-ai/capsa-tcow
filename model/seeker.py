'''
Neural network architecture description.
Created by Basile Van Hoorick for TCOW.
'''

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'seeker/'))

from __init__ import *

# Internal imports.
import mask_tracker


class Seeker(torch.nn.Module):

    def __init__(self, logger,wrapper=None,wrapper_arg=None, **kwargs):
        super().__init__()
        self.logger = logger
        self.wrapper = wrapper
        self.seeker = mask_tracker.QueryMaskTracker(logger, **kwargs)
        self.wrapper_arg = wrapper_arg

        if self.wrapper_arg == 'sculpt':
            self.forward = self.sculpt_forward
        elif self.wrapper_arg == 'vote':
            self.forward = self.vote_forward
        elif self.wrapper_arg == 'sample':
            self.forward = self.sample_forward


    def wrap(self):
        self.wrapped_seeker = self.wrapper(self.seeker) if self.wrapper != None else None

    def forward(self, *args):
        return self.seeker(*args)

    def sculpt_forward(self, *args):
        return self.wrapped_seeker(*args,return_risk=False)

    def vote_forward(self, *args):
        return self.wrapped_seeker(*args,return_risk=False,tile_and_reduce=False)

    def sample_forward(self, *args):
        return self.wrapped_seeker(*args,return_risk=False)



