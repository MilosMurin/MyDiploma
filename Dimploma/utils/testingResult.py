import os
from abc import ABC, abstractmethod
import numpy as np
import torch
from matplotlib import pyplot as plt
import pandas as pd

from Dimploma import util
from Dimploma.utils.my_agent_base import MyAgent

from scipy.stats import pearsonr

class TestBase(ABC):
    def __init__(self, node_amount, test_amount=100):
        self.test_amount = test_amount
        self.node_amount = node_amount
        self.agent_names = []
        self.agent_colors = []
        self.agents = []
        self.agent_multiple_tests = []
        self.agent_special = []

    def addAgent(self, name, agent: MyAgent, color='blue', multiple_tetst=False, special=False):
        self.agent_names.append(name)
        self.agents.append(agent)
        self.agent_colors.append(color)
        self.agent_multiple_tests.append(multiple_tetst)
        self.agent_special.append(special)

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def test(self, env, special = None):
        pass

class TestResult(TestBase):
    def __init__(self, node_amount, test_amount=100):
        super().__init__(node_amount, test_amount)
        self.objs = np.zeros(test_amount, dtype=np.float32)
        self.rews = np.zeros(test_amount, dtype=np.float32)
        self.actions = np.zeros((test_amount, node_amount * 2), dtype=np.int16)

    def setup(self):
        ags = len(self.agents)
        self.objs = torch.zeros((ags, self.test_amount), dtype=torch.float32)
        self.rews = torch.zeros((ags, self.test_amount), dtype=torch.float32)
        self.actions = torch.zeros((ags, self.test_amount, self.node_amount * 2), dtype=torch.int16)

    def test(self, env, special=False):
        for j, agent in enumerate(self.agents):
            if self.agent_special[j] == special:
                if self.agent_multiple_tests[j]:
                    for i in range(self.test_amount):
                        obj, sel, rew, acts = agent.test(env, argmax=False)
                        self.objs[j, i] = obj
                        self.rews[j, i] = rew[-1]
                        self.actions[j, i, :len(acts)] = torch.tensor(acts)
                else:
                    obj, sel, rew, acts = agent.test(env)
                    self.objs[j, :] = obj
                    self.rews[j, :] = rew[-1]
                    self.actions[j, :, :len(acts)] = torch.tensor(acts)


    def print_result(self, rews=False):
        for j in range(len(self.agents)):
            print(self.agent_names[j])
            print(f'Objs: Mean: {self.objs[j].mean():.2f}, Min: {self.objs[j].min():.2f}, Max: {self.objs[j].max():.2f}')
            if rews:
                print(f'Rews: Mean: {self.rews[j].mean():.2f}, Min: {self.rews[j].min():.2f}, Max: {self.rews[j].max():.2f}')
        plt.bar(self.agent_names, self.objs.mean(axis=1), color=self.agent_colors)

    def print_result_pretty(self):
        for j in range(len(self.agents)):
            print(self.agent_names[j])
            self.print_with_test(j)
            # print(f'Priemer: {self.objs[j].mean():.2f}, Min: {self.objs[j].min():.2f}, Max: {self.objs[j].max():.2f}')
        plt.bar(self.agent_names, self.objs.mean(axis=1), color=self.agent_colors)


    def print_with_test(self, a_index, test_equality=True):
        mean_r = self.objs[a_index].mean()
        min_r = self.objs[a_index].min()
        max_r = self.objs[a_index].max()
        if test_equality and (mean_r == min_r and mean_r == max_r):
            print(f'Priemer, Min, Max: {mean_r:.2f}')
        else:
            print(f'Priemer: {mean_r:.2f}, Min: {min_r:.2f}, Max: {max_r:.2f}')


class TestCorrelResult(TestBase):
    def __init__(self, node_amount, test_amount=100, graph_amount=100, name='test', append=-1):
        super().__init__(node_amount, test_amount)
        self.graph_amount = graph_amount
        self.append = append
        # self.header = ['graph'] + list(range(node_amount))
        self.header = ['graph', 'correlation']
        self.name = name + f'_t{self.test_amount}'
        self.path = os.path.join('results/correl/', f'{self.name}/')
        self.deg_path = os.path.join(self.path, 'degrees.csv')
        self.agent_paths = []

    def addAgent(self, name, agent: MyAgent, color='blue', multiple_tetst=False, special=False):
        super().addAgent(name, agent, color, multiple_tetst, special)
        self.agent_paths.append(os.path.join('results/', f'{self.name}/{name.replace(" ", "_").lower()}_actions.csv'))

    def setup(self, name='test'):
        self.name = name + f'_n{self.node_amount}_t{self.test_amount}'
        self.path = os.path.join('results/correl/', f'{self.name}/')
        self.deg_path = os.path.join(self.path, 'degrees.csv')
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(f'{self.path}/graphs/', exist_ok=True)
        self.agent_paths = []
        if self.append == -1:
            pd.DataFrame(columns=self.header).to_csv(self.deg_path, index=False)
        for i, agent_name in enumerate(self.agent_names):
            self.agent_paths.append(os.path.join(self.path, f'{agent_name.replace(" ", "_").lower()}_actions.csv'))
            if self.append == -1:
                pd.DataFrame(columns=self.header).to_csv(self.agent_paths[i], index=False)
                pd.DataFrame(columns=self.header).to_csv(self.agent_paths[i], index=False)

    def test(self, env, special=None, argmax=False):
        print(f'Started tests')
        for g in range(self.graph_amount):
            gi = g + (self.append if self.append > 0 else 0)
            gr, _ = env.reset(True)
            degrees = util.get_out_edges(gr)
            # writ = torch.cat([torch.tensor([gi]), degrees])
            # df = pd.DataFrame([writ.tolist()], columns=self.header)
            # df.to_csv(self.deg_path, mode='a', header=False, index=False)
            torch.save(gr, f'{self.path}graphs/graph{gi}')
            print(f'Graph {gi}------------------------------')
            for j, agent in enumerate(self.agents):
                print(f'Started tests for agent {self.agent_names[j]}')
                # actions = torch.zeros(self.test_amount, self.node_amount, dtype=torch.int32)
                writ = torch.zeros(2, dtype=torch.float32)
                writ[0] = gi
                if self.agent_special[j]:
                    logits = agent.test_correl(special, reset_graph=False)
                else:
                    logits = agent.test_correl(env, reset_graph=False)
                corr = pearsonr(degrees, logits.squeeze())
                writ[1] = corr[0]
                # print(f'Finished all tests for agent {self.agent_names[j]}')
                df = pd.DataFrame([writ.tolist()], columns=self.header)
                df.to_csv(self.agent_paths[j], mode='a', header=False, index=False)
            print(f'Finished {gi + 1} tests for all agents')
        print(f'Ended tests')