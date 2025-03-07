from ast import List
import datetime
import math
import numpy as np
from utils.utils import write_to_file

class MovingAverageScore:
    def __init__(self, count = 100):
        self.memory = np.zeros(count)

        self.index = 0
        self.memory_size = count
        self.full_memory = False

        self.best_avg_score = - math.inf
        self.best_score = - math.inf
        self.count_of_episodes = 0

    def add(self, scores):
        length = len(scores)
        if length > 0:
            if length > self.memory_size:
                scores = scores[-self.memory_size:]
                length = self.memory_size

            scores = np.array(scores)
            self.best_score = max(self.best_score, scores.max())
            self.count_of_episodes += length

            if length + self.index <= self.memory_size:
                new_index = self.index + length
                self.memory[self.index:new_index] = scores

                if new_index == self.memory_size:
                    self.index = 0
                    self.full_memory = True
                else:
                    self.index = new_index
            else:
                length_to_end = self.memory_size - self.index
                length_from_start = length - length_to_end

                self.memory[self.index:] = scores[:length_to_end]
                self.memory[:length_from_start] = scores[length_to_end:]

                self.index = length_from_start
                self.full_memory = True

    def mean(self):
        if self.full_memory:
            mean = self.memory.mean()
        else:
            if self.index == 0:
                return -math.inf, False
            mean = self.memory[:self.index].mean()

        if self.best_avg_score < mean:
            self.best_avg_score = mean
            return mean, True
        return mean, False

    def get_best_avg_score(self):
        return self.best_avg_score

    def get_count_of_episodes(self):
        return self.count_of_episodes

class Logger:
    def __init__(self, file_name: str, columns: list = []) -> None:
        self.log_data = ','.join(columns)
        self.file_name = file_name
        
    def log(self, *columns):
        self.log_data += '\n' + ','.join(list(map(str, columns)))

    def save(self):
        write_to_file(self.log_data, self.file_name)

class AgentLogger(Logger):
    def __init__(self, file_name: str, columns: List = [], save_every = 10) -> None:
        super().__init__(file_name, ['time', 'iteration', 'episode'] + columns)
        self.save_every = save_every

    def log(self, iteration, episode, *columns):
        super().log(datetime.datetime.now(), iteration, episode, *columns)

        if iteration % self.save_every == 0:
            self.save()

class ScoreLogger(Logger):
    def __init__(self, file_name: str, save_every = 10, print_every = 1, score_transformer_fn = None):
        super().__init__(file_name, ['iteration', 'episode', 'avg_reward', 'best_avg_reward', 'best_reward'] + (['avg_score', 'best_avg_score'] if score_transformer_fn != None else []))
        self.save_every = save_every
        self.print_every = print_every
        self.score_transformer_fn = score_transformer_fn
        self.mva = MovingAverageScore()

    def log(self, iteration, rewards, infos):
        self.mva.add(rewards)
        episode = self.mva.get_count_of_episodes()
        avg_reward, got_better = self.mva.mean()

        avg_score, best_avg_score, best_score = None, None, None
        if self.score_transformer_fn != None:
            avg_score = self.score_transformer_fn(avg_reward)
            best_avg_score = self.score_transformer_fn(self.mva.best_avg_score)
            best_score = self.score_transformer_fn(self.mva.best_score)
            super().log(iteration, episode, avg_reward, self.mva.best_avg_score, self.mva.best_score, avg_score, best_avg_score)
        else:
            super().log(iteration, episode, avg_reward, self.mva.best_avg_score, self.mva.best_score)

        if iteration % self.save_every == 0:
            self.save()

        if iteration % self.print_every == 0:
            print(f'Iteration {iteration}\tepisode {episode}\tavg score {avg_reward if self.score_transformer_fn is None else avg_score:.5f}'
                  f'\tbest score {self.mva.best_score if self.score_transformer_fn is None else best_score:.5f}'
                  f'\tbest avg score {self.mva.best_avg_score if self.score_transformer_fn is None else best_avg_score:.5f}')

        return episode, avg_reward, got_better, self.mva.best_avg_score