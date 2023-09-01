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

    def __init__(self, logger,wrapper=None, **kwargs):
        super().__init__()
        self.logger = logger
        self.wrapper = wrapper
        self.seeker = mask_tracker.QueryMaskTracker(logger, **kwargs)

    def wrap(self):
        self.wrapped_seeker = self.wrapper(self.seeker) if self.wrapper != None else None

    def forward(self, *args):
        return self.seeker(*args)

    def wrapped_forward(self, *args):
        assert self.wrapped_seeker!=None , "Please pass a wrapper when initializing the seeker."
        return self.wrapped_seeker(*args,return_risk=True)



