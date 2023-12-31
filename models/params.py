import os

learning_rate = 5e-4      # learning rate for adam optimizer
buffer_size = int(1e6)    # size of the replay buffer
batch_size = 256          # Minibatch size for each gradient update
tau = 0.005               # (float) – the soft update coefficient (“Polyak update”, between 0 and 1)
gamma = 0.99              # (float) – the discount factor
total_timesteps=int(2e8)  # (int) – The total number of samples (env steps) to train on
policy_interval=int(2e4)  # (int) – Policy saving interval
Checkpoint_interval=int(4e5)  # (int) – Checkpoint saving interval
max_episodes=int(1e5)     # (int) – Maximum number of episodes to run

eval_freq = int(500)  # 训练过程中每隔多少个step进行一次模型评估
n_eval_episodes = 2   # 训练过程中模型评估时的episode局数

FRAMES_NUM = 2  # 2 or 3

if FRAMES_NUM == 3:
    StateDim = (42, )  # 状态维度
else:
    StateDim = (48, )  # 状态维度 24*2

ActionDim = (4, )   # 动作维度
Train = 2      # 0，1，2分别表示：0重新训练，1加载之前的训练，2测试模式


class FolderPath:
    def __init__(self, role='red'):
        self.role = role
        save_date = '2023_08/2023_08_03'
        load_date = '2023_08/2023_08_03'
        self.load_steps = 3600000
        model_name = "sac_model_"+str(self.load_steps)+"_steps.zip"
        flag_use_checkpoints = False
        # Red
        if role == 'red':
            self.Save_ModelDir_Red = "./models/Red/"+save_date
            self.Load_ModelDir_Red = "./models/Red/" + load_date
            self.LogDir_Red = "./logs/Red/"+save_date
            saveDir = self.Save_ModelDir_Red
            loadDir = self.Load_ModelDir_Red
            logDir = self.LogDir_Red
        else:
        # Blue
            self.Save_ModelDir_Blue = "./models/Blue/"+save_date
            self.Load_ModelDir_Blue = "./models/Blue/"+load_date
            self.LogDir_Blue = "./logs/Blue/"+save_date
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
        if not os.path.exists(self.loadModelDir) and Train != 0:
            print("加载模型路径错误！ 路径：" + self.loadModelDir)



