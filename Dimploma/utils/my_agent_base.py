from abc import ABC, abstractmethod

import numpy as np
import torch

class MyAgent(ABC):
    @abstractmethod
    def test(self, env, argmax=True, reset_graph=True):
        pass

    @abstractmethod
    def test_correl(self, env, reset_graph=True):
        pass


class RandomAgent(MyAgent):
    def test(self, env, argmax=True, reset_graph=True):
        terminal = False
        observation, mask = env.reset(reset_graph)
        rewards = []
        actions_res = []
        masks_res = []
        while not terminal:
            action = np.random.choice(np.arange(mask.shape[0])[mask])

            masks_res.append(mask)
            observation, mask, reward, terminal, _ = env.step(action)
            rewards.append(reward)
            actions_res.append(action)

        return env.compute_objective_function(), env.graph.edge_attr[:, 1], rewards, actions_res  # , masks_res

    def test_correl(self, env, reset_graph=True):
        observation, mask = env.reset(reset_graph)
        node_amount = observation.x.shape[0]
        return torch.fill(torch.zeros(node_amount), 1/node_amount)

class OptimalAgent(MyAgent):
    def test(self, env, argmax=True, reset_graph=True):
        return env.calculate_min_span_tree(), [], [1], []  # , masks_res

    def test_correl(self, env, reset_graph=True):
        observation, mask = env.reset(reset_graph)
        node_amount = observation.x.shape[0]
        return torch.zeros(node_amount)