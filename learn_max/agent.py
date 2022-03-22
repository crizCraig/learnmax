import dataclasses
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class AgentState:
    # TODO: Perhaps this should be a PyTorch module as it just holds a bunch of tensors
    state: torch.Tensor = None  # sensor state returned by env TODO(OOM): Don't populate this when only training GPT as it's not needed
    dvq_x: torch.Tensor = None  # state preprocessed for dvq
    dvq_x_hat: torch.Tensor = None
    dvq_z_q_emb: torch.Tensor = None
    dvq_z_q_flat: torch.Tensor = None  # same as dvq_z_q_emb just flattened for use as single token
    dvq_z_q_ind: torch.Tensor = None
    dvq_latent_loss: torch.Tensor = None
    dvq_recon_loss: torch.Tensor = None
    dvq_loss: torch.Tensor = None
    append_i: torch.tensor = -1  # for bug hunting
    split: torch.tensor = 0  # 0 for train 1 for test

    def dict(self):
        return self.__dict__

    def to(self, device):
        for field in dataclasses.fields(self):
            val = getattr(self, field.name)
            if torch.is_tensor(val):
                setattr(self, field.name, val.to(device))


class LearnMaxAgent:
    """ Learn max agent that takes most interesting action using prediction uncertainty
        This object conforms with pytorch lightning bolts conventions on agents in RL setups.
        It's really just a dynamics function right now which maps from external states to internal states and actions.
    """
    # TODO: Move this all to model.py and delete the agent class.
    def __init__(self, model,
                 num_actions: int,  # Number of discrete actions available
                 num_environments: int = 1,
                 num_search_steps: int = 10,  # Number of steps to beam search into predicted trajectories
                 ):
        self.model = model
        self.num_actions = num_actions
        # self.s_buff = deque(maxlen=model.gpt_block_size)  # History of states
        # self.a_buff = deque(maxlen=model.gpt_block_size)  # History of actions
        # Start with noop:
        #  https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/docs/manual/manual.pdf
        # self.a_buff.append([0] * num_environments)

        self.num_environments = num_environments

    @torch.no_grad()
    def get_action(self, agent_state: torch.Tensor, device: str) -> Tuple[List[int], AgentState]:
        """
        Takes in the current state and returns the action based on the agents policy

        Args:
            agent_state: current state of the environment and internal states (i.e. dvq states)
            device: the device used for the current batch

        Returns:
            action defined by policy
            predicted_trajectory of action-states considered most likely when choosing above action
        """
            # print(f'states {device=}')

        # Ways to measure uncertainty / interestingness
        # - We have a deviation head on the transformer that tries to just learn this from prob changes over time.
        # - We can also group predicted next a_z by the action - using a dict to map z's to actions, then
        #   we can pursue actions that have the lowest variance - i.e. we are least confident about their next state
        #   This has the problem that some actions are just inherently stochastic. So it'd be addictive the way gambling
        #   is addictive to humans. This as opposed to the above, where we've already seen that uncertainty decreases
        #   over time even in the presence of random training data.
        # - Compare the current window with the masked predictions for intermediate steps in the window, this gives us
        #   the current uncertainty for the recent past. This is more useful for OOD / anomaly detection saying, okay
        #   I'm in a really uncommon trajectory - stop, get help, or lower learning rate. This doesn't allow beam searching
        #   for forward uncertainty though.
        # - Another way is a brute force - trajectory counting approach. The longer such trajectories are, the bigger
        #   the more keys a count dictionary would contain and the lower their constituent counts would be. This can
        #   be used to check the above heuristics. Also psuedo-count methods cf Marc G Bellmare
        # - DVQ reconstruction loss - if the image hasn't been seen before, we will do a really bad job at reconstructing,
        #   esp. if it's a new level in zuma for example
        dvq_x = agent_state.dvq_x
        if not self.model.should_train_gpt:
            raise NotImplementedError('Need to revive this code')
            # TODO: We have already forwarded these through the model, so there's no reason to re-forward. We just
            #   need to compute the average loss and run the manual backward.
            # TODO: Move self.dvq_ready setting to LearnMax model
            self.s_buff.append(dvq_x)
            dvq_batch_ready = self.model.training and len(self.s_buff) == self.s_buff.maxlen
            if dvq_batch_ready:
                dvq_x = torch.cat(tuple(self.s_buff))  # Stack states in 0th (batch) dimension
                self.s_buff.clear()
                self.dvq_ready = True  # dvq outputs are now based on some training

        # Return a random action if we haven't filled buffer of z states.
        if len(self.model.train_buf) < self.model.gpt_block_size:  # TODO: Use self.dvq_ready to do dvq training again
            ret = self.get_random_action(len(dvq_x))
            predicted_trajectory = None
        else:
            # Search through tree of predicted a,z to find most interesting future
            # TODO: Use @no_train for below when moving to LearnMax
            was_training = self.model.training
            self.model.eval()
            ret, predicted_trajectory = self.model.tree_search()
            if was_training:
                self.model.train()

        # # get the logits and pass through softmax for probability distribution
        # probabilities = F.softmax(self.net(states)).squeeze(dim=-1)
        # prob_np = probabilities.data.cpu().numpy()
        #
        # # take the numpy values and randomly select action based on prob distribution
        # actions = [np.random.choice(len(prob), p=prob) for prob in prob_np]

        # return actions
        # self.a_buff.append(ret)

        if 'GO_RIGHT_AGENT' in os.environ:
            ret = [3]
        return ret, predicted_trajectory

    def get_random_action(self, num: int) -> List[int]:
        """returns a random action"""
        actions = []

        for i in range(num):
            action = np.random.randint(0, self.num_actions)
            actions.append(action)

        return actions
