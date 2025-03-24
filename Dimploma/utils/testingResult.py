import numpy as np
from matplotlib import pyplot as plt

from Dimploma.utils.my_agent_base import MyAgent


class TestResult:
    def __init__(self, node_amount, test_amount=100):
        self.test_amount = test_amount
        self.node_amount = node_amount
        self.agent_names = []
        self.agent_colors = []
        self.agents = []
        self.objs = np.zeros(test_amount, dtype=np.float32)
        self.rews = np.zeros(test_amount, dtype=np.float32)
        self.actions = np.zeros((test_amount, node_amount * 2), dtype=np.int16)

    def addAgent(self, name, agent: MyAgent, color='blue'):
        self.agent_names.append(name)
        self.agents.append(agent)
        self.agent_colors.append(color)

    def test(self, env):
        ags = len(self.agents)
        self.objs = np.zeros((ags, self.test_amount), dtype=np.float32)
        self.rews = np.zeros((ags, self.test_amount), dtype=np.float32)
        self.actions = np.zeros((ags, self.test_amount, self.node_amount * 2), dtype=np.int16)
        for i in range(self.test_amount):
            for j, agent in enumerate(self.agents):
                obj, sel, rew, acts = agent.test(env)
                self.objs[j, i] = obj
                self.rews[j, i] = rew[-1]
                self.actions[j, i, :len(acts)] = acts


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