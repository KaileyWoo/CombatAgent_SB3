from os import path, makedirs

learning_rate = 5e-4      # learning rate for adam optimizer
buffer_size = int(1e6)    # size of the replay buffer
batch_size = 256          # Minibatch size for each gradient update
tau = 0.005               # (float) – the soft update coefficient (“Polyak update”, between 0 and 1)
gamma = 0.99              # (float) – the discount factor
total_timesteps=int(2e8)  # (int) – The total number of samples (env steps) to train on
max_episodes=int(1e5)     # (int) – Maximum number of episodes to run

n_eval_episodes = 5   # 训练过程中模型评估时的episode局数
test_timesteps = int(2e8)  # 测试的步数

StateDim = (48, )  # 状态维度 24*2
ActionDim = (4, )   # 动作维度

class Params(object):
    def __init__(self, role='red', num_envs=1, test_episodes=500, load_model=""):
        self.role = role
        self.test_episodes = test_episodes
        self.policy_interval=int(2e4)  # (int) – Policy saving interval
        self.Checkpoint_interval = int(4e5)  # (int) – Checkpoint saving interval
        self.eval_freq = int(500)  # 训练过程中每隔多少个step进行一次模型评估
        if num_envs > 1:
            self.policy_interval = max(self.policy_interval // num_envs, 1)
            self. Checkpoint_interval = max(self.Checkpoint_interval // num_envs, 1)
            self.eval_freq = max(self.eval_freq // num_envs, 1)
        # 红方
        if self.role == 'red':
            self.Train = 2  # 0，1，2分别表示：0重新训练，1加载之前的训练，2测试模式
            # 文件夹相关路径处理
            save_date = '2023_08/2023_08_22'
            load_date = '2023_08/2023_08_09'
            flag_use_checkpoints = False
            self.load_steps = 20800000
            model_name = "sac_model_" + str(self.load_steps) + "_steps.zip"

            self.Save_ModelDir_Red = "./save_models/Red/" + save_date
            self.Load_ModelDir_Red = "./save_models/Red/" + load_date
            self.LogDir_Red = "./save_logs/Red/" + save_date
            saveDir = self.Save_ModelDir_Red
            loadDir = self.Load_ModelDir_Red
            logDir = self.LogDir_Red

        # 蓝方
        else:
            self.Train = 2
            # 文件夹相关路径处理
            save_date = '2023_08/2023_08_13'
            load_date = '2023_08/2023_08_09'
            flag_use_checkpoints = False
            self.load_steps = 20800000
            model_name = "sac_model_" + str(self.load_steps) + "_steps.zip"

            self.Save_ModelDir_Blue = "./save_models/Blue/" + save_date
            self.Load_ModelDir_Blue = "./save_models/Red/" + load_date
            self.LogDir_Blue = "./save_logs/Blue/" + save_date
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

        if self.Train != 2:
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
        else:
            self.loadModelDir = load_model

        self.load_flag = True
        if not path.exists(self.loadModelDir) and self.Train != 0:
            print("错误！加载模型路径错误，路径：" + self.loadModelDir)
            self.load_flag = False








