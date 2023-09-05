import os

learning_rate = 5e-4      # learning rate for adam optimizer
buffer_size = int(1e6)    # size of the replay buffer
batch_size = 256          # Minibatch size for each gradient update
tau = 0.005               # (float) �C the soft update coefficient (��Polyak update��, between 0 and 1)
gamma = 0.99              # (float) �C the discount factor
total_timesteps=int(2e8)  # (int) �C The total number of samples (env steps) to train on
policy_interval=int(2e4)  # (int) �C Policy saving interval
Checkpoint_interval=int(4e5)  # (int) �C Checkpoint saving interval
max_episodes=int(1e5)     # (int) �C Maximum number of episodes to run

num_envs = 1   # ��������(���������)

eval_freq = int(500)  # ѵ��������ÿ�����ٸ�step����һ��ģ������
n_eval_episodes = 5   # ѵ��������ģ������ʱ��episode����

test_episodes = 500  # ���ԵĻغ���
test_timesteps = int(2e8)  # ���ԵĲ���

if num_envs > 1:
    policy_interval = max(policy_interval // num_envs, 1)
    Checkpoint_interval = max(Checkpoint_interval // num_envs, 1)
    eval_freq = max(eval_freq // num_envs, 1)

StateDim = (48, )  # ״̬ά�� 24*2
ActionDim = (4, )   # ����ά��


class Params(object):
    def __init__(self, role='red'):
        self.role = role
        # �췽
        if self.role == 'red':
            self.Train = 2  # 0��1��2�ֱ��ʾ��0����ѵ����1����֮ǰ��ѵ����2����ģʽ
            # �ļ������·������
            save_date = '2023_08/2023_08_22'
            load_date = '2023_08/2023_08_09'
            flag_use_checkpoints = False
            self.load_steps = 20800000
            model_name = "sac_model_" + str(self.load_steps) + "_steps.zip"

            self.Save_ModelDir_Red = "./models/Red/" + save_date
            self.Load_ModelDir_Red = "./models/Red/" + load_date
            self.LogDir_Red = "./logs/Red/" + save_date
            saveDir = self.Save_ModelDir_Red
            loadDir = self.Load_ModelDir_Red
            logDir = self.LogDir_Red

        # ����
        else:
            self.Train = 2
            # �ļ������·������
            save_date = '2023_08/2023_08_13'
            load_date = '2023_08/2023_08_09'
            flag_use_checkpoints = False
            self.load_steps = 20800000
            model_name = "sac_model_" + str(self.load_steps) + "_steps.zip"

            self.Save_ModelDir_Blue = "./models/Blue/" + save_date
            self.Load_ModelDir_Blue = "./models/Red/" + load_date
            self.LogDir_Blue = "./logs/Blue/" + save_date
            saveDir = self.Save_ModelDir_Blue
            loadDir = self.Load_ModelDir_Blue
            logDir = self.LogDir_Blue

        self.CheckpointDir = saveDir + "/sac_checkpoints/"
        self.modelDir = saveDir + "/sac_save_model/"
        self.bestModelDir = saveDir + "/sac_best_model/"
        if flag_use_checkpoints:
            self.loadModelDir = loadDir + "/sac_checkpoints/" + model_name
        else:
            self.loadModelDir = loadDir + "/sac_save_model/models.zip"
        self.tensorboardDir = logDir + '/sac_tensorboard/'
        self.monitorDir = logDir + '/sac_monitor/'
        self.evalDir = logDir + '/sac_eval/'

        if not os.path.exists(self.CheckpointDir):
            os.makedirs(self.CheckpointDir)
        if not os.path.exists(self.modelDir):
            os.makedirs(self.modelDir)
        if not os.path.exists(self.bestModelDir):
            os.makedirs(self.bestModelDir)
        if not os.path.exists(self.tensorboardDir):
            os.makedirs(self.tensorboardDir)
        if not os.path.exists(self.monitorDir):
            os.makedirs(self.monitorDir)
        if not os.path.exists(self.evalDir):
            os.makedirs(self.evalDir)
        if not os.path.exists(self.loadModelDir) and self.Train != 0:
            print("����ģ��·������ ·����" + self.loadModelDir)








