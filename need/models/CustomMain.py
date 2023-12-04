from stable_baselines3 import SAC#, PPO, A2C, DDPG
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize  # DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback  # StopTrainingOnMaxEpisodes,
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.monitor import Monitor
# from gymnasium import make
from need.models.params import *
from need.models.MyCallback import MyCallback
# from models.CallbackAfterEval import CallbackAfterEval
# import env
from need.env.MyCombatEnv import MyCombatEnv
from time import time
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau


def make_env(rank, config_dic={}):
    def _init():
        my_env = MyCombatEnv(config_dic=config_dic, env_id=rank)
        return my_env
    return _init


def CustomMain(config):
    # 读取配置项
    config_dict = {}
    config_dict["host"] = str(config.get('database', 'host'))
    config_dict["port"] = int(config.getint('database', 'port'))
    config_dict["role"] = str(config.get('database', 'role'))
    config_dict["role_id"] = str(config.get('database', 'role_id'))
    #num_envs = int(config.get('database', 'num_envs'))
    num_envs = 10
    load_model = str(config.get('database', 'load_model'))
    monitor_dir = str(config.get('database', 'monitor_dir'))
    episodes = int(config.get('database', 'episodes'))
    is_lunch = config.get('database', 'launch_missile')
    if is_lunch.lower() == "true":
        config_dict["is_lunch"] = True
    else:
        config_dict["is_lunch"] = False

    # 定义相关参数
    params = Params(role=config_dict["role"], num_envs=num_envs, test_episodes=episodes, load_model=load_model, test_monitor_dir=monitor_dir)

    if params.Train != 0 and not params.load_success:
        return

    # 创建多环境
    print("共创建 {} 组环境".format(num_envs))
    # envs = [make_env(env_id='MyCombatEnv-v0', rank=i, role=role, role_id=role_id, host=host, port=port) for i in range(num_envs)]
    envs = [make_env(rank=i, config_dic=config_dict) for i in range(num_envs)]
    combatEnv = SubprocVecEnv(envs)
    # combatEnv = VecNormalize(combatEnv, norm_obs=True, norm_reward=True,clip_obs=10.)   #标准化

    if params.Train < 2:
        # 训练
        if params.monitorDir:
            combatEnv = VecMonitor(combatEnv, params.monitorDir)

        # 定义模型
        model = SAC('MlpPolicy', combatEnv, verbose=0, device='cuda', tensorboard_log=params.tensorboardDir, gradient_steps=-1,
                    batch_size=batch_size, learning_rate=learning_rate, buffer_size=buffer_size)

        # 定义回调函数 ---begin---
        # 保存模型的中间状态
        # checkpoint_callback = CheckpointCallback(save_freq=params.Checkpoint_interval, save_path=params.CheckpointDir, name_prefix='sac_model',
        #                                           save_replay_buffer=True, save_vecnormalize=False)
        # 自定义回调函数（1.定时保存策略；2.若需训练过程中进行模型评估，则设置环境的评估模式标志位）
        my_callback = MyCallback(policy_interval=params.policy_interval, save_dir=params.modelDir, best_save_dir=params.bestModelDir,
                                 monitor_dir=params.monitorDir, load_dir=params.loadModelDir, eval_freq=0)
        # 在训练过程中进行模型性能评估
        eval_callback = EvalCallback(combatEnv, best_model_save_path=params.bestModelDir, log_path=params.evalDir, n_eval_episodes=params.n_eval_episodes,
                                      eval_freq=params.eval_freq, deterministic=True, verbose=0)
        # 达到最大指定的回合数时提前终止训练
        #callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=max_episodes, verbose=1)
        # 学习率回调函数
        #scheduler = ExponentialLR(optimizer=model.policy.optimizer, gamma=0.99)
        #scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=10)
        callback = [my_callback]
        # 定义回调函数 ---end---

        if params.Train == 0:
            print("重头开始训练模型！ 模型保存路径：" + params.modelDir)
            print("日志保存路径：" + params.tensorboardDir)
            model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
        elif params.Train == 1:
            print("加载模型，路径：" + params.loadModelDir + "models.zip")
            model.set_parameters(params.loadModelDir + "models.zip")
            # model = SAC.load(params.loadModelDir + "models.zip")
            print("继续训练模型！ 模型保存路径：" + params.modelDir)
            print("日志保存路径：" + params.tensorboardDir)
            model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False, progress_bar=True)

    else:
        # 测试
        if params.test_monitor_dir:
            combatEnv = VecMonitor(combatEnv, params.test_monitor_dir)

        if params.Train == 2:
            print("测试模型，共测试 " + str(params.test_episodes) + " 回合！")
            print("加载模型，路径：" + params.loadModelDir)
            model = SAC.load(params.loadModelDir)
            # model = MlpPolicy.load(params.loadModelDir)

            obs = combatEnv.reset()
            win = 0
            draw = 0
            total_reward = 0
            episode = 0
            start_time = time()
            for step in range(test_timesteps):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, info = combatEnv.step(action)
                for env_idx in range(num_envs):
                    if terminated[env_idx]:
                        episode += 1
                        total_reward += info[env_idx]['episode']['r']
                        if config_dict["role"] == "red" and info[env_idx]['result'] == 1 or config_dict[
                            "role"] == "blue" and info[env_idx]['result'] == 2:
                            win += 1
                        if info[env_idx]['result'] == 3:
                            draw += 1

                if episode >= params.test_episodes:
                    break
            end_time = time()
            execution_time = end_time - start_time
            print("***********************************************************************")
            print("测试结束！共测试 " + str(episode) + " 回合！胜利 " + str(win) + " 回合！平局" + str(draw) + "回合！")
            print("胜率：" + str(win / episode))
            print("平均奖励：" + str(total_reward / episode))
            print(f"Execution time: {execution_time / 60:.2f} min ({execution_time:.2f} s)")

        elif params.Train == 3:
            print("估计策略模型，共估计 " + str(params.test_episodes) + " 回合！")
            print("加载模型，路径：" + params.loadModelDir)
            model = MlpPolicy.load(params.loadModelDir)
            start_time = time()
            mean_reward, std_reward = evaluate_policy(model, combatEnv, n_eval_episodes=params.test_episodes, deterministic=True)
            end_time = time()
            execution_time = end_time - start_time
            print("***********************************  估计结束  ************************************")
            print("平均奖励：" + str(mean_reward))
            print(f"Execution time: {execution_time / 60:.2f} min ({execution_time:.2f} s)")







