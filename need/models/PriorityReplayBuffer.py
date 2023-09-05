from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np


class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, obs_shape, action_shape, device, alpha=0.6, beta=0.4, epsilon=1e-6):
        super(PriorityReplayBuffer, self).__init__(buffer_size, obs_shape, action_shape, device)
        self.alpha = alpha  # 控制优先级采样的程度，通常取值为0.6
        self.beta = beta  # 控制重要性采样权重的程度，通常取值为0.4
        self.epsilon = epsilon  # 用于避免优先级为0时采样概率为0的情况
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.sample_idx = 0

    def add(self, obs, next_obs, action, reward, done, info):
        # 添加经验样本，并更新优先级为初始值
        super(PriorityReplayBuffer, self).add(obs, next_obs, action, reward, done, info)
        self.priorities[self.sample_idx] = self.epsilon  # 初始优先级设置为epsilon
        self.sample_idx = (self.sample_idx + 1) % self.buffer_size

    def update_priorities(self, idxs, priorities):
        # 更新样本的优先级
        self.priorities[idxs] = priorities

    def get_priority(self, td_errors):
        # 计算样本的优先级
        return np.abs(td_errors) + self.epsilon

    def sample(self, batch_size: int, beta=0.4) -> dict:
        # 计算优先级采样概率
        priorities = self.priorities[:self.size]
        priority_probabilities = priorities ** self.alpha
        priority_probabilities /= priority_probabilities.sum()

        # 根据优先级概率抽样
        batch_indices = np.random.choice(self.size, batch_size, p=priority_probabilities)
        batch = self.sample_batch(batch_indices)

        # 计算重要性采样权重
        weights = (self.size * priority_probabilities[batch_indices]) ** (-self.beta)
        weights /= weights.max()

        # 将重要性采样权重添加到batch中
        batch['weights'] = np.array(weights, dtype=np.float32)

        return batch














