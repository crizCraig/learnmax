from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from pl_bolts.models.rl.common.agents import Agent

from learn_max.dvq.vqvae import VQVAE
from learn_max.mingpt.model import GPT


class LearnMaxAgent:
    """ Learn max agent that takes most interesting action using prediction uncertainty """

    def __init__(self, dvq: VQVAE, gpt: GPT, num_search_steps=10):
        self.dvq = dvq
        self.gpt = gpt
        self.z_buff = deque(maxlen=gpt.block_size)  # History of states for transformer

    @torch.no_grad()
    def __call__(self, states: torch.Tensor, prev_actions: torch.Tensor, device: str) -> List[int]:
        """
        Takes in the current state and returns the action based on the agents policy

        Args:
            states: current state of the environment
            prev_actions: previous actions taken that led to `states`
            device: the device used for the current batch

        Returns:
            action defined by policy
        """
        if not isinstance(states, list):
            states = [states]

        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, device=device)
        z_buff = self.z_buff

        # TODO: Ensure that we pass an action state to each Env and aren't passing actions to one env and states to
        #   another, etc...
        action_states = torch.cat((prev_actions, states), 0)

        # Get compressed / quantized action-state representation from discrete auto-encoder
        x_hat, z_q_emb, latent_loss, z_q_ind = self.dvq(action_states)
        z_buff.append(z_q_emb)

        # Ways to measure uncertainty / interestingness
        # - We have a deviation head on the transformer that tries to just learn this from prob changes over time
        # - We can also group predicted next z(a,s) by the action - using a dict to map z's to actions, then
        #   we can pursue actions that have the lowest variance - i.e. we are least confident about their next state
        #   This has the problem that some actions are just inherently stochastic. So it'd be addictive the way gambling
        #   is addictive to humans. This as opposed to the above, where we've already seen that uncertainty decreases
        #   over time even in the presence of random training data.
        # - Another way is a brute force - trajectory counting approach. The longer such trajectories are, the bigger
        #   the more keys a count dictionary would contain and the lower their constituent counts would be. This can
        #   be used to check the above heuristics.

        # If we haven't filled buffer of actions, just return no-op
        if len(z_buff) < z_buff.maxlen:
            return noops()




        # TODO: Search through tree of predicted action states to find most interesting future
        #   Then forward that z=(a,s) through the decoder to get the action



        # # get the logits and pass through softmax for probability distribution
        # probabilities = F.softmax(self.net(states)).squeeze(dim=-1)
        # prob_np = probabilities.data.cpu().numpy()
        #
        # # take the numpy values and randomly select action based on prob distribution
        # actions = [np.random.choice(len(prob), p=prob) for prob in prob_np]

        return actions
