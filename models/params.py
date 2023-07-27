import os

learning_rate = 5e-4      # learning rate for adam optimizer
buffer_size = int(1e6)    # size of the replay buffer
batch_size = 256          # Minibatch size for each gradient update
tau = 0.005               # (float) – the soft update coefficient (“Polyak update”, between 0 and 1)
gamma = 0.99              # (float) – the discount factor
total_timesteps=int(2e8)  # (int) – The total number of samples (env steps) to train on
policy_interval=int(4e5)  # (int) – Policy saving interval
max_episodes=int(1e5)     # (int) – Maximum number of episodes to run

eval_freq = int(1000) # 每隔多少个step进行一次测试
n_eval_episodes = 2 # 测试时的episode数

FRAMES_NUM = 2  # 2 or 3

if FRAMES_NUM == 3:
    StateDim = (42, )  # 状态维度
else:
    StateDim = (15, )  # 状态维度

ActionDim = (4, )   # 动作维度
Train = 0      # 0，1，2分别表示：0重新训练，1加载之前的训练，2测试模式

class FolderPath:
    def __init__(self):
        # Red
        save_date = '2023_07_27'
        load_date = '2023_07_16'
        model_name = "sac_model_2000000_steps.zip"
        self.Save_ModelDir_Red = "./models/Red/"+save_date
        self.Load_ModelDir_Red = "./models/Red/"+load_date+"/sac_checkpoints/"+model_name
        self.LogDir_Red = "./logs/Red/"+save_date
        # Blue
        # self.Save_ModelDir_Blue = "./models/Blue/"+save_date+"/sac_checkpoints/"
        # self.Load_ModelDir_Blue = "./models/Blue/"+load_date+"/sac_checkpoints/"+model_name
        # self.LogDir_Blue = "./logs/Blue/"+save_date+"/"

        self.create_folders()

    def create_folders(self):
        if not os.path.exists(self.Save_ModelDir_Red):
            os.makedirs(self.Save_ModelDir_Red)
        if not os.path.exists(self.Load_ModelDir_Red) and Train != 0:
            print("加载模型路径错误！ 路径：" + self.Load_ModelDir_Red)
        if not os.path.exists(self.LogDir_Red):
            os.makedirs(self.LogDir_Red)

        # if not os.path.exists(self.Save_ModelDir_Blue):
        #     os.makedirs(self.Save_ModelDir_Blue)
        # if not os.path.exists(self.Load_ModelDir_Blue):
        #     os.makedirs(self.Load_ModelDir_Blue)
        # if not os.path.exists(self.LogDir_Blue):
        #     os.makedirs(self.LogDir_Blue)

