from stable_baselines3 import PPO, A2C, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from models.params import *
from logs.MyCallback import MyCallback
import env


def main(role='red', role_id='1001'):
    # 定义相关文件路径
    folderPath = FolderPath()
    saveDir = folderPath.Save_ModelDir_Red
    loadDir = folderPath.Load_ModelDir_Red
    logDir = folderPath.LogDir_Red
    if role == 'blue':
        saveDir = folderPath.Save_ModelDir_Blue
        loadDir = folderPath.Load_ModelDir_Blue
        logDir = folderPath.LogDir_Blue
    tensorboardDir = logDir+'sac_tensorboard/'
    monitorDir = logDir+'sac_monitor/'
    # 创建环境
    env = gym.make('MyCombatEnv-v0', role=role, role_id=role_id)
    env = Monitor(env, monitorDir, allow_early_resets=True, override_existing=False)
    env = DummyVecEnv([lambda: env])
    # 定义模型
    model = SAC('MlpPolicy', env, verbose=0, device='cuda', tensorboard_log=tensorboardDir,
                batch_size=batch_size, learning_rate=learning_rate, buffer_size=buffer_size)
    # 定义回调函数
    checkpoint_callback = CheckpointCallback(save_freq=policy_interval, save_path=saveDir, name_prefix='sac_model')
                                             # save_replay_buffer=True, save_vecnormalize=True)
    #callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=max_episodes, verbose=1)
    callback = [checkpoint_callback, MyCallback()]
    # 训练或测试模型
    if Train != 0:
        print("加载模型，路径：" + loadDir)
        model.load(loadDir)
    if Train != 2:
        print("训练模型！ 模型保存路径：" + saveDir)
        print("日志保存路径：" + tensorboardDir)
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    else:
        print("测试模型......")
        obs, info = env.reset()
        for step in range(100):
            action, _states = model.predict(obs, deterministic=True)
            print("Step {}".format(step + 1))
            obs, reward, terminated, truncated, info = env.step(action)
            print('obs=', obs, '\n reward=', reward, '  done=', terminated or truncated)
            if terminated or truncated:
                obs, info = env.reset()