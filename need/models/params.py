from os import path, makedirs

learning_rate = 5e-4      # learning rate for adam optimizer
buffer_size = int(1e6)    # size of the replay buffer
batch_size = 256          # Minibatch size for each gradient update
tau = 0.005               # (float) – the soft update coefficient (“Polyak update”, between 0 and 1)
gamma = 0.99              # (float) – the discount factor
total_timesteps=int(2e8)  # (int) – The total number of samples (env steps) to train on
max_episodes=int(1e5)     # (int) – Maximum number of episodes to run

test_timesteps = int(2e8)  # 测试的步数

StateDim = (48, )  # 状态维度 24*2
ActionDim = (4, )   # 动作维度

class Params(object):
    def __init__(self, role='red', num_envs=1, test_episodes=500, load_model="", test_monitor_dir=""):
        self.role = role
        self.test_episodes = test_episodes
        self.policy_interval=int(2e4)  # (int) – Policy saving interval
        self.Checkpoint_interval = int(4e5)  # (int) – Checkpoint saving interval
        self.eval_freq = int(2e5)  # 训练过程中每隔多少个step进行一次模型评估
        self.n_eval_episodes = 10 * num_envs   # 训练过程中模型评估时的episode局数
        if num_envs > 1:
            self.policy_interval = max(self.policy_interval // num_envs, 1)
            self. Checkpoint_interval = max(self.Checkpoint_interval // num_envs, 1)
            self.eval_freq = max(self.eval_freq // num_envs, 1)

        self.Train = 2  # 0，1，2分别表示：0重新训练，1加载之前的训练，2测试模式, 3估计策略
        # 文件夹相关路径处理
        save_date = '2023_12/2023_12_02_3000'
        load_date = '2023_12/2023_12_01'
        saveDir = "./save_models/Red/"
        loadDir = "./save_models/Red/"
        logDir = "./save_logs/Red/"

        self.CheckpointDir = saveDir + save_date + "/sac_checkpoints/"
        self.modelDir = saveDir + save_date + "/sac_save_model/"
        self.bestModelDir = saveDir + save_date + "/sac_best_model/"

        self.tensorboardDir = logDir + save_date + '/sac_tensorboard/'
        self.monitorDir = logDir + save_date + '/sac_monitor/'
        self.evalDir =logDir + save_date + '/sac_eval/'

        self.loadModelDir = loadDir + load_date + "/sac_save_model/"

        if self.Train < 2:
            if not path.exists(self.CheckpointDir):
                makedirs(self.CheckpointDir)
            if not path.exists(self.modelDir):
                makedirs(self.modelDir)
            if not path.exists(self.bestModelDir):
                makedirs(self.bestModelDir)
            if not path.exists(self.tensorboardDir):
                makedirs(self.tensorboardDir)
            if not path.exists(self.monitorDir):
                makedirs(self.monitorDir)
            if not path.exists(self.evalDir):
                makedirs(self.evalDir)

            self.load_success = True
            if not path.exists(self.loadModelDir + "models.zip") and self.Train != 0:
                print("错误！加载模型路径错误，路径：" + self.loadModelDir + "models.zip")
                self.load_success = False
        else:
            self.test_monitor_dir = test_monitor_dir
            if not path.exists(self.test_monitor_dir):
                makedirs(self.test_monitor_dir)

            self.loadModelDir = load_model  # 测试模式从文件读取路径
            self.load_success = True
            if not path.exists(self.loadModelDir) and self.Train != 0:
                print("错误！加载模型路径错误，路径：" + self.loadModelDir)
                self.load_success = False









