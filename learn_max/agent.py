from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from pl_bolts.models.rl.common.agents import Agent


class LearnMaxAgent(Agent):
    """ Learn max agent that takes most interesting action using prediction uncertainty """

    @torch.no_grad()
    def __call__(self, states: torch.Tensor, device: str) -> List[int]:
        """
        Takes in the current state and returns the action based on the agents policy

        Args:
            states: current state of the environment
            device: the device used for the current batch

        Returns:
            action defined by policy
        """
        if not isinstance(states, list):
            states = [states]

        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, device=device)

        # TODO: Search through tree of predicted action states to find most interesting future
        #   Then forward that z=(a,s) through the decoder to get the action

        # # get the logits and pass through softmax for probability distribution
        # probabilities = F.softmax(self.net(states)).squeeze(dim=-1)
        # prob_np = probabilities.data.cpu().numpy()
        #
        # # take the numpy values and randomly select action based on prob distribution
        # actions = [np.random.choice(len(prob), p=prob) for prob in prob_np]

        return actions
