
import torch

from Dimploma import util
from Dimploma.EnvironmentTree import EnvMinimalTreeTwoStep, EnvMaximalTreeTwoStep, EnvInfo, EnvMinimalTreeTwoStepHeur
from Dimploma.ppo_parallel_custom import Agent


def load_agent_for_testing(path, iteration_amount, env_info: EnvInfo, gcn, device='cpu', maximal=False, pth=None):
    testesing_agent = path
    testing_iter_amount = iteration_amount
    if pth is None:
        testing_last_path = f'/models/iter_{testing_iter_amount - 1}_last.pt'
    else:
        testing_last_path = pth
    util.plot_training(testesing_agent)
    loaded_graph = torch.load(testesing_agent + '/graph.pt', map_location=device, weights_only=False)
    if loaded_graph is not None:
        util.show_data(loaded_graph)
        env_info.graph_provider.set_fixed_graph(loaded_graph)

    if maximal:
        test_env = EnvMaximalTreeTwoStep(env_info)
    else:
        test_env = EnvMinimalTreeTwoStep(env_info)

    agent_test = Agent(model=gcn, device=device, name=path + "_test", override=True, test=True)

    agent_test.load_model(f"{testesing_agent}{testing_last_path}")

    return loaded_graph, test_env, agent_test


def load_desc(path):
    with open(path + '/desc.txt', 'r') as file:
        content = file.read()
    print(content)


