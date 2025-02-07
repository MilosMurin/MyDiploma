import json
from datetime import datetime
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
from torch.multiprocessing import Process, Pipe
from torch_geometric.data import Batch, Data
from utils.logger import AgentLogger, ScoreLogger
from utils.utils import flatten_list, write_to_file


# import ipdb
def test_random(env):
    terminal = False
    observation, mask = env.reset()
    rewards = []
    actions_res = []
    masks_res = []
    while not terminal:
        action = np.random.choice(np.arange(mask.shape[0])[mask])

        masks_res.append(mask)
        observation, mask, reward, terminal, _ = env.step(action)
        rewards.append(reward)
        actions_res.append(action)

    return env.compute_objective_function(), observation.edge_attr[:, 1], rewards, actions_res #, masks_res

# 1 proces
# Odohranie hry v niekolkych prostrediach
def worker(connection, env_params, env_func, count_of_iterations, count_of_envs,
           count_of_steps, gamma, gae_lambda, index):
    sys.stdout.flush()
    # ipdb.set_trace()
    envs = [env_func(*env_params, process_i=index, env_i=e) for e in range(count_of_envs)]
    observations, masks = list(map(list, zip(*[env.reset() for env in envs])))
    # print(f'Observations x shape: {observations[0].x.shape}')
    # print(f'Observations edge shape: {observations[0].edge_index.shape}')
    masks = torch.stack(masks)
    # print(f'Masks shape: {masks.shape}')
    game_rewards = np.zeros(count_of_envs)

    for o in observations:
        o.cpu()

    mem_observations = np.empty((count_of_steps, count_of_envs), dtype=Data)
    mem_masks = torch.zeros((count_of_steps, count_of_envs, masks.shape[1]))
    mem_log_probs = torch.zeros((count_of_steps, count_of_envs, 1))
    mem_actions = torch.zeros((count_of_steps, count_of_envs, 1), dtype=torch.long)
    mem_values = torch.zeros((count_of_steps + 1, count_of_envs, 1))
    mem_rewards = torch.zeros((count_of_steps, count_of_envs, 1))

    for iteration in range(count_of_iterations):
        mem_non_terminals = torch.ones((count_of_steps, count_of_envs, 1))
        scores = []
        env_infos = []

        # Hranie prostredia
        for step in range(count_of_steps):
            # print("start step env", envs[0].graph.edge_attr)
            connection.send(observations)  # 1 A
            # print("after send", envs[0].graph.edge_attr)
            logits, values, rewards = connection.recv()  # 2 B
            # print(f'Logits shape: {logits.shape}')
            # print(f'Values shape: {values.shape}')
            logits = torch.where(masks.cpu(), logits.cpu(), torch.tensor(-1e+8).cpu())
            # print(f'Logits shape after: {logits.shape}')
            probs = F.softmax(logits, dim=-1)
            # print(f'Probs shape: {probs.shape}')
            actions = probs.multinomial(num_samples=1)
            log_probs = F.log_softmax(logits, dim=-1).gather(1, actions)

            mem_observations[step] = observations
            mem_masks[step] = masks
            mem_log_probs[step] = log_probs
            mem_actions[step] = actions
            mem_values[step] = values
            mem_rewards[step] = rewards

            game_rewards += rewards.squeeze(-1).numpy()
            infos = []

            # Vykonanie jedneho kroku v kazdom prostredi v ramci workera
            for idx in range(count_of_envs):
                # print("start step", envs[idx].graph.edge_attr)
                observation, mask, reward, terminal, info = envs[idx].step(actions[idx, 0].item())
                mem_rewards[step, idx, 0] = reward
                game_rewards[idx] += reward
                infos.append(info)
                # if reward < 0:
                #    mem_non_terminals[step, idx, 0] = 0

                if terminal:
                    mem_non_terminals[step, idx, 0] = 0
                    scores.append(game_rewards[idx])
                    game_rewards[idx] = 0
                    observation, mask = envs[idx].reset()
                observations[idx] = observation
                masks[idx] = mask
                # print("end step", envs[idx].graph.edge_attr)

            env_infos.append(infos)

        connection.send(observations)  # 3 A
        mem_values[step + 1] = connection.recv()  # 4 B

        '''
            Values - calculating advantage using gae
        '''
        # mem_rewards = torch.clamp(mem_rewards, -1.0, 1.0)
        advantages = torch.zeros((count_of_steps, count_of_envs, 1))
        values = torch.zeros((count_of_steps, count_of_envs, 1))
        gae = torch.zeros((count_of_envs, 1))

        for step in reversed(range(count_of_steps)):
            delta = mem_rewards[step] + gamma * mem_values[step + 1] * mem_non_terminals[step] \
                    - mem_values[step]
            gae = delta + gamma * gae_lambda * gae * mem_non_terminals[step]
            values[step] = gae + mem_values[step]
            advantages[step] = gae.clone()

        connection.send([mem_observations, mem_masks, mem_log_probs, mem_actions, values,
                         advantages, scores, env_infos])  # 5 A
    connection.recv()
    connection.close()


class Agent:
    def __init__(self, model, rnd_model=None, gamma=0.99, epsilon=0.1,
                 coef_value=0.5, coef_entropy=0.001, gae_lambda=0.95,
                 name='ppo', path='results/', device='cpu', lr=0.00025, override=False, test=False, early_stop=False):

        self.model = model
        self.rnd_model = rnd_model
        self.model.to(device)
        # self.rnd_model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.rnd_optimizer = torch.optim.Adam(self.rnd_model.parameters(), lr=lr)

        self.gamma = gamma
        self.coef_value = coef_value
        self.coef_entropy = coef_entropy
        self.gae_lambda = gae_lambda

        self.lower_bound = 1 - epsilon
        self.upper_bound = 1 + epsilon

        self.name = name

        self.path = os.path.join(path, f'{datetime.now().strftime("%y%m%d_%H#")}{name}/')

        if not test:
            if not override and os.path.exists(self.path):
                raise Exception(f'The training {name} already exists')
            else:
                os.makedirs(self.path, exist_ok=True)

        self.device = device
        self.train_desc = None
        self.finish_training = False
        self.last_saved = datetime.now()

        self.early_stop = early_stop
        self.early_val = -1

    def training_description(self, description):
        self.train_desc = description
        write_to_file(description, f'{self.path}/desc.txt')

    def test(self, env):
        terminal = False
        observation, mask = env.reset()
        rewards = []
        actions_res = []
        masks_res = []
        while not terminal:
            with torch.no_grad():
                logits, values = self.model(observation.to(self.device))

            logits = torch.where(mask.cpu(), logits.cpu(), torch.tensor(-1e+8).cpu())
            # print(f'Logits shape after: {logits.shape}')
            probs = F.softmax(logits, dim=-1)
            # print(f'Probs shape: {probs.shape}')
            actions = probs.multinomial(num_samples=1)

            masks_res.append(mask)
            observation, mask, reward, terminal, _ = env.step(actions.item())
            rewards.append(reward)
            actions_res.append(actions.item())

        return env.compute_objective_function(), observation.edge_attr[:, 1], rewards, actions_res#, masks_res
        # print(f'Obj function: {env.compute_objective_function()} State: {observation.x[:, 1]} '
        #       f'Reward: {rewards}')

    def train(self, env_params, env_func, count_of_actions,
              count_of_iterations=1000, count_of_processes=2,
              count_of_envs=16, count_of_steps=128, count_of_epochs=4,
              batch_size=512, score_transformer_fn=None):

        print('Training is starting')
        self.finish_training = False
        self.early_val = -1

        loss_logger = AgentLogger(f'{self.path}/loss.csv',
                                  ['avg_score', 'policy', 'value', 'entropy', 'rnd', 'lr'])
        score_logger = ScoreLogger(f'{self.path}/score.csv', score_transformer_fn=score_transformer_fn)

        lr_scheduler = LinearLR(self.optimizer, start_factor=1, end_factor=0.01,
                                total_iters=int(count_of_iterations / 2))
        buffer_size = count_of_processes * count_of_envs * count_of_steps
        batches_per_iteration = count_of_epochs * buffer_size / batch_size

        processes, connections = [], []
        for p in range(count_of_processes):
            parr_connection, child_connection = Pipe()
            process = Process(target=worker, args=(
                child_connection, env_params, env_func, count_of_iterations,
                count_of_envs, count_of_steps, self.gamma, self.gae_lambda, p))
            connections.append(parr_connection)
            processes.append(process)
            # print("starting process")
            process.start()

        for iteration in range(count_of_iterations):
            print(f"Iteration {iteration} starting")
            if self.finish_training:
                break

            for step in range(count_of_steps):
                observations = [conn.recv() for conn in connections]  # 1 B

                with torch.no_grad():
                    observations = flatten_list(observations)
                    observations = Batch.from_data_list(observations).to(self.device)
                    logits, values = self.model(observations)

                    '''
                    RND - Rewards
                    '''
                    # rnd_pred, rnd_targ = self.rnd_model(observations)

                    # rewards = (rnd_pred - rnd_targ) ** 2
                    # rewards = rewards.sum(dim=1) / 2
                    rewards = torch.zeros((count_of_processes * count_of_envs))

                # If you selected actions in the main process, your iteration
                # would last about 0.5 seconds longer (measured on 2 processes)
                logits = logits.view(count_of_processes, -1, count_of_actions).cpu()
                # logits = logits.view(count_of_processes, count_of_envs, -1, count_of_actions).cpu()
                values = values.view(count_of_processes, -1, 1).cpu()
                rewards = rewards.view(count_of_processes, -1, 1).cpu()

                for idx in range(count_of_processes):
                    connections[idx].send((logits[idx], values[idx], rewards[idx]))  # 2 A

            observations = [conn.recv() for conn in connections]  # 3 B
            observations = flatten_list(observations)
            observations = Batch.from_data_list(observations).to(self.device)

            with torch.no_grad():
                _, values = self.model(observations)
                values = values.view(count_of_processes, -1, 1).cpu()

            for conn_idx in range(count_of_processes):
                connections[conn_idx].send((values[conn_idx]))  # 4 A

            (mem_observations, mem_masks, mem_log_probs, mem_actions,
             mem_target_values, mem_advantages, end_games,
             env_infos) = [], [], [], [], [], [], [], []

            for connection in connections:
                (observations, masks, log_probs, actions, target_values, advantages,
                 score_of_end_games, infos) = connection.recv()  # 5 B

                with open(os.path.join(self.path, 'scores_debug.csv'), 'a+') as f:
                    f.write(f'{iteration}, "{json.dumps(score_of_end_games)}"\n ')

                mem_observations.extend(observations.flatten())
                mem_masks.append(masks)
                mem_actions.append(actions)
                mem_log_probs.append(log_probs)
                mem_target_values.append(target_values)
                mem_advantages.append(advantages)
                end_games.extend(score_of_end_games)
                env_infos.extend(infos)

            episode, avg_score, better_score = score_logger.log(iteration, end_games, env_infos)

            mem_observations = Batch.from_data_list(mem_observations)
            mem_masks = torch.stack(mem_masks).bool().view(-1, count_of_actions)
            mem_actions = torch.stack(mem_actions).view(-1, 1)
            mem_log_probs = torch.stack(mem_log_probs).view(-1, 1)
            mem_target_values = torch.stack(mem_target_values).view(-1, 1)
            mem_advantages = torch.stack(mem_advantages).view(-1, 1)
            mem_advantages = (mem_advantages - torch.mean(mem_advantages)) / (torch.std(mem_advantages) + 1e-5)

            s_policy, s_value, s_entropy, s_rnd = 0, 0, 0, 0

            # Learning here
            for epoch in range(count_of_epochs):
                perm = torch.randperm(buffer_size).view(-1, batch_size)
                for idx in perm:
                    obs = Batch.from_data_list(mem_observations[idx]).to(self.device)
                    logits, values = self.model(obs)
                    logits = torch.where(mem_masks[idx].to(self.device), logits,
                                         torch.tensor(-1e+8, device=self.device))
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    new_log_probs = log_probs.gather(1, mem_actions[idx].to(self.device))

                    entropy_loss = (log_probs * probs).sum(1, keepdim=True).mean()
                    value_loss = F.mse_loss(values, mem_target_values[idx].to(self.device))

                    # rnd_pred, rnd_targ = self.rnd_model(obs)
                    # loss_rnd = (rnd_targ - rnd_pred) ** 2

                    # random loss regularisation, 25% non zero for 128envs, 100% non zero for 32envs
                    # prob = 16.0 / (count_of_envs * count_of_processes)
                    # random_mask = torch.rand(loss_rnd.shape).to(loss_rnd.device)
                    # random_mask = 1.0 * (random_mask < prob)
                    # loss_rnd = (loss_rnd * random_mask).sum() / (random_mask.sum() + 0.00000001)

                    ratio = torch.exp(new_log_probs - mem_log_probs[idx].to(self.device))
                    advantage = mem_advantages[idx].to(self.device)
                    surr_policy = ratio * advantage
                    surr_clip = torch.clamp(ratio, self.lower_bound, self.upper_bound) * advantage
                    policy_loss = - torch.min(surr_policy, surr_clip).mean()

                    s_policy += policy_loss.item()
                    s_value += value_loss.item()
                    s_entropy += entropy_loss.item()
                    # s_rnd += loss_rnd.item()

                    self.optimizer.zero_grad()
                    loss = policy_loss + self.coef_value * value_loss \
                           + self.coef_entropy * entropy_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

                    # self.rnd_optimizer.zero_grad()
                    # loss_rnd.backward()
                    # torch.nn.utils.clip_grad_norm_(self.rnd_model.parameters(), 0.5)
                    # self.rnd_optimizer.step()
                # if epoch % 10 == 0:
                #     print(f'Epoch: {epoch}')

            lr_scheduler.step()

            loss_logger.log(iteration, episode, avg_score,
                            s_policy / batches_per_iteration,
                            s_value / batches_per_iteration,
                            s_entropy / batches_per_iteration,
                            s_rnd / batches_per_iteration,
                            self.optimizer.param_groups[0]['lr'])

            mins = (datetime.now() - self.last_saved).seconds // 60

            if self.early_stop and iteration % 250 == 0:
                if self.early_val == -1:
                    self.early_val = avg_score
                    print('Writing to early val:', self.early_val)
                else:
                    print('Comparing early val:', self.early_val, ' - ', avg_score)
                    if self.early_val + 0.01 > avg_score:
                        print('Stopping early because not many improvements were made')
                        break
                    self.early_val = avg_score

            if better_score or iteration % 1000 == 0 or mins >= 30:
                self.save_model(iteration)
                self.last_saved = datetime.now()

        print('Training has ended, best avg score is ', score_logger.mva.get_best_avg_score())
        if self.train_desc is not None:
            print('Desc:', self.train_desc)

        self.save_model(f'{iteration-1}_early_stop')
        self.save_model(f'{count_of_iterations-1}_last')

        for connection in connections:
            connection.send(1)
        for process in processes:
            process.join()

    def stop_training(self):
        print('Stoping the training')
        self.finish_training = True

    def save_model(self, name):
        os.makedirs(f'{self.path}/models/', exist_ok=True)
        torch.save(self.model.state_dict(), f'{self.path}/models/iter_{str(name)}.pt')

    def load_model(self, path):
        print('Loading model from', path)
        self.model.load_state_dict(torch.load(path))
