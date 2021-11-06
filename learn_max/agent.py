import os
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

@dataclass
class AgentState:
    dvq_x: torch.Tensor = None
    dvq_x_hat: torch.Tensor = None
    dvq_z_q_emb: torch.Tensor = None
    dvq_z_q_flat: torch.Tensor = None # same as dvq_z_q_emb just flattened for use as single token
    dvq_z_q_ind: torch.Tensor = None
    dvq_latent_loss: torch.Tensor = None
    dvq_recon_loss: torch.Tensor = None
    dvq_loss: torch.Tensor = None

    def dict(self):
        return self.__dict__


class LearnMaxAgent:
    """ Learn max agent that takes most interesting action using prediction uncertainty
        This object conforms with pytorch lightning bolts conventions on agents in RL setups.
        It's really just a dynamics function right now which maps from external states to internal states and actions.
    """

    def __init__(self, model,
                 num_actions: int,  # Number of discrete actions available
                 num_environments: int = 1,
                 num_search_steps: int = 10,  # Number of steps to beam search into predicted trajectories
                 ):
        self.model = model
        self.num_actions = num_actions
        self.num_search_steps = num_search_steps
        self.s_buff = deque(maxlen=model.gpt_block_size)  # History of states
        self.a_buff = deque(maxlen=model.gpt_block_size)  # History of actions

        # Start with noop:
        #  https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/docs/manual/manual.pdf
        self.a_buff.append([0] * num_environments)

        self.num_environments = num_environments

        # Training stuff
        self.dvq_ready = False

    @torch.no_grad()
    def __call__(self, states: torch.Tensor, device: str) -> Tuple[List[int], AgentState]:
        """
        Takes in the current state and returns the action based on the agents policy

        Args:
            states: current state of the environment
            device: the device used for the current batch

        Returns:
            action defined by policy and AgentState (agent's internal state)
        """
        states, agent_states = states  # Internal and external states

        if not isinstance(states, list):
            states = [states]

        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, device=device)  # causes issues when num_workers > 0
            # print(f'states {device=}')

        # TODO: Ensure that we pass an action state to each Env and aren't passing actions to one env and states to
        #   another, etc...

        # Ways to measure uncertainty / interestingness
        # - We have a deviation head on the transformer that tries to just learn this from prob changes over time.
        # - We can also group predicted next z(a,s) by the action - using a dict to map z's to actions, then
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
        #   be used to check the above heuristics.
        # - DVQ reconstruction loss - if the image hasn't been seen before, we will do a really bad job at reconstructing,
        #   esp. if it's a new level in zuma for example

        dvq_x = states

        if not self.model.training_gpt:
            # TODO: We have already forwarded these through the model, so there's no reason to re-forward. We just
            #   need to compute the average loss and run the manual backward.
            self.s_buff.append(states)
            dvq_batch_ready = self.model.training and len(self.s_buff) == self.s_buff.maxlen
            if dvq_batch_ready:
                dvq_x = torch.cat(tuple(self.s_buff))  # Stack states in 0th (batch) dimension
                self.s_buff.clear()
                self.dvq_ready = True  # dvq outputs are now based on some training

        x, x_hat, z_q_emb, z_q_flat, latent_loss, recon_loss, dvq_loss, z_q_ind = self.model.dvq(
            dvq_x, wait_to_init=not self.dvq_ready)

        agent_state = AgentState(dvq_x=x, dvq_x_hat=x_hat, dvq_z_q_emb=z_q_emb, dvq_z_q_flat=z_q_flat,
                                 dvq_latent_loss=latent_loss, dvq_recon_loss=recon_loss, dvq_loss=dvq_loss,
                                 dvq_z_q_ind=z_q_ind)

        # Return a random action if we haven't filled buffer of z states.
        if True or not dvq_batch_ready:
            ret = self.get_random_action(states)
        else:
            # Search through tree of predicted z,a to find most interesting future
            ret = self.model.beam_search(z_q_flat, self.a_buff)

        # # get the logits and pass through softmax for probability distribution
        # probabilities = F.softmax(self.net(states)).squeeze(dim=-1)
        # prob_np = probabilities.data.cpu().numpy()
        #
        # # take the numpy values and randomly select action based on prob distribution
        # actions = [np.random.choice(len(prob), p=prob) for prob in prob_np]

        # return actions
        self.a_buff.append(ret)

        return ret, agent_state

    def get_random_action(self, state: torch.Tensor) -> List[int]:
        """returns a random action"""
        actions = []

        for i in range(len(state)):
            action = np.random.randint(0, self.num_actions)
            actions.append(action)

        return actions
