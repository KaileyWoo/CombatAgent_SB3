from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np


class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, obs_shape, action_shape, device, alpha=0.6, beta=0.4, epsilon=1e-6):
        super(PriorityReplayBuffer, self).__init__(buffer_size, obs_shape, action_shape, device)
        self.alpha = alpha  # �������ȼ������ĳ̶ȣ�ͨ��ȡֵΪ0.6
        self.beta = beta  # ������Ҫ�Բ���Ȩ�صĳ̶ȣ�ͨ��ȡֵΪ0.4
        self.epsilon = epsilon  # ���ڱ������ȼ�Ϊ0ʱ��������Ϊ0�����
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.sample_idx = 0

    def add(self, obs, next_obs, action, reward, done, info):
        # ��Ӿ������������������ȼ�Ϊ��ʼֵ
        super(PriorityReplayBuffer, self).add(obs, next_obs, action, reward, done, info)
        self.priorities[self.sample_idx] = self.epsilon  # ��ʼ���ȼ�����Ϊepsilon
        self.sample_idx = (self.sample_idx + 1) % self.buffer_size

    def update_priorities(self, idxs, priorities):
        # �������������ȼ�
        self.priorities[idxs] = priorities

    def get_priority(self, td_errors):
        # �������������ȼ�
        return np.abs(td_errors) + self.epsilon

    def sample(self, batch_size: int, beta=0.4) -> dict:
        # �������ȼ���������
        priorities = self.priorities[:self.size]
        priority_probabilities = priorities ** self.alpha
        priority_probabilities /= priority_probabilities.sum()

        # �������ȼ����ʳ���
        batch_indices = np.random.choice(self.size, batch_size, p=priority_probabilities)
        batch = self.sample_batch(batch_indices)

        # ������Ҫ�Բ���Ȩ��
        weights = (self.size * priority_probabilities[batch_indices]) ** (-self.beta)
        weights /= weights.max()

        # ����Ҫ�Բ���Ȩ����ӵ�batch��
        batch['weights'] = np.array(weights, dtype=np.float32)

        return batch














