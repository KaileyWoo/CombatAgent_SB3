from stable_baselines3 import PPO, A2C, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnMaxEpisodes, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from models.params import *
from models.MyCallback import MyCallback
from models.CallbackAfterEval import CallbackAfterEval
import env
import time


def make_env(env_id, rank, role='red', role_id='1001'):
    def _init():
        my_env = gym.make(env_id, role=role, role_id=role_id, port=8868+rank)
        return my_env
    return _init

def main(role='red', role_id='1001'):
    # 定义相关文件路径
    folders = FolderPath(role=role)

    # 创建多环境
    envs = [make_env(env_id='MyCombatEnv-v0', rank=i, role=role, role_id=role_id) for i in range(num_envs)]
    combatEnv = SubprocVecEnv(envs)
    combatEnv = VecMonitor(combatEnv, folders.monitorDir)

    # 创建单环境
    # combatEnv = gym.make('MyCombatEnv-v0', role=role, role_id=role_id, port=8868)
    # combatEnv = Monitor(combatEnv, folders.monitorDir, allow_early_resets=True, override_existing=False)
    # combatEnv = DummyVecEnv([lambda: combatEnv])

    # 定义模型
    model = SAC('MlpPolicy', combatEnv, verbose=0, device='cuda', tensorboard_log=folders.tensorboardDir,
                batch_size=batch_size, learning_rate=learning_rate, buffer_size=buffer_size)
    # model = PPO('MlpPolicy', combatEnv, verbose=0, device='cuda', tensorboard_log=tensorboardDir,
    #             batch_size=batch_size, learning_rate=learning_rate, buffer_size=buffer_size)

    # 定义回调函数 ---begin---
    # 保存模型的中间状态
    checkpoint_callback = CheckpointCallback(save_freq=Checkpoint_interval, save_path=folders.CheckpointDir, name_prefix='sac_model')
                                              #save_replay_buffer=True, save_vecnormalize=True)
    # 自定义回调函数（1.定时保存策略；2.若需训练过程中进行模型评估，则设置环境的评估模式标志位）
    my_callback = MyCallback(policy_interval=policy_interval, save_dir=folders.modelDir, env=combatEnv, eval_freq=0)
    # 在训练过程中进行模型性能评估
    # eval_callback = EvalCallback(combatEnv, best_model_save_path=folders.bestModelDir, log_path=folders.evalDir, n_eval_episodes=n_eval_episodes,
    #                               callback_after_eval=CallbackAfterEval(), eval_freq=eval_freq, deterministic=True)
    # 达到最大指定的回合数时提前终止训练
    #callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=max_episodes, verbose=1)
    callback = [checkpoint_callback, my_callback]
    # 定义回调函数 ---end---

    # 训练或测试模型
    if Train != 0:
        print("加载模型，路径：" + folders.loadModelDir)
        model.set_parameters(folders.loadModelDir)
        #model.load(loadModelDir)  # 会重新创建一个全新的模型
    if Train == 0:
        print("重头开始训练模型！ 模型保存路径：" + folders.modelDir)
        print("日志保存路径：" + folders.tensorboardDir)
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    elif Train == 1:
        print("继续训练模型！ 模型保存路径：" + folders.modelDir)
        print("日志保存路径：" + folders.tensorboardDir)
        #steps = total_timesteps - folders.load_steps
        model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False, progress_bar=True)
    else:
        test_episodes = 100   # 测试的回合数
        test_timesteps = 100000   # 测试的步数
        print("测试模型，共测试 " + str(test_episodes) + " 回合！")
        obs = combatEnv.reset()
        win = 0
        total_reward = 0
        episodes = 0
        start_time = time.time()
        for step in range(test_timesteps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = combatEnv.step(action)
            for env_idx in range(num_envs):
                if terminated[env_idx]:
                    episodes += 1
                    total_reward += info[env_idx]['episode']['r']
                    if info[env_idx]['done'] == 1:
                        win += 1
            if episodes >= test_episodes:
                break
        end_time = time.time()
        execution_time = end_time - start_time
        print("***********************************************************************")
        print("测试结束！共测试 " + str(episodes) + " 回合！胜利 " + str(win) + " 回合！")
        print("胜率：" + str(win / test_episodes))
        print("平均奖励：" + str(total_reward / test_episodes))
        print(f"Execution time: {execution_time:.6f} seconds")


