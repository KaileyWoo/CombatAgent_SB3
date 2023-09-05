from stable_baselines3 import SAC  # PPO, A2C, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv  # , VecMonitor  # DummyVecEnv
# from stable_baselines3.common.callbacks import CheckpointCallback  # StopTrainingOnMaxEpisodes, EvalCallback
# from stable_baselines3.common.monitor import Monitor
# from gymnasium import make
from need.models.params import *
# from need.models.MyCallback import MyCallback
# from models.CallbackAfterEval import CallbackAfterEval
# import env
from need.env.MyCombatEnv import MyCombatEnv
from time import time


def make_env(rank, role='red', role_id='1001', host='127.0.0.1', port=8868):
    def _init():
        my_env = MyCombatEnv(role=role, role_id=role_id, env_id=rank, host=host, port=port)
        return my_env
    return _init


def CustomMain(config):
    # 读取配置项
    host = str(config.get('database', 'host'))
    port = int(config.getint('database', 'port'))
    role = str(config.get('database', 'role'))
    role_id = str(config.get('database', 'role_id'))
    num_envs = int(config.get('database', 'num_envs'))
    load_model = str(config.get('database', 'load_model'))
    episodes = int(config.get('database', 'episodes'))

    # 定义相关参数
    params = Params(role=role, num_envs=num_envs, test_episodes=episodes, load_model=load_model)

    if not params.load_flag:
        return

    # 创建多环境
    print("共创建 {} 组环境".format(num_envs))
    # envs = [make_env(env_id='MyCombatEnv-v0', rank=i, role=role, role_id=role_id, host=host, port=port) for i in range(num_envs)]
    envs = [make_env(rank=i, role=role, role_id=role_id, host=host, port=port) for i in range(num_envs)]
    combatEnv = SubprocVecEnv(envs)
    # if not params.monitorDir:
    #     combatEnv = VecMonitor(combatEnv, params.monitorDir)

    # 定义模型
    model = SAC('MlpPolicy', combatEnv, verbose=0, device='cuda', tensorboard_log=params.tensorboardDir,
                batch_size=batch_size, learning_rate=learning_rate, buffer_size=buffer_size)

    # if params.Train != 2:
    #     # 定义回调函数 ---begin---
    #     # 保存模型的中间状态
    #     checkpoint_callback = CheckpointCallback(save_freq=params.Checkpoint_interval, save_path=params.CheckpointDir, name_prefix='sac_model')
    #                                               #save_replay_buffer=True, save_vecnormalize=True)
    #     # 自定义回调函数（1.定时保存策略；2.若需训练过程中进行模型评估，则设置环境的评估模式标志位）
    #     my_callback = MyCallback(policy_interval=params.policy_interval, save_dir=params.modelDir, env=combatEnv, eval_freq=0)
    #     # 在训练过程中进行模型性能评估
    #     # eval_callback = EvalCallback(combatEnv, best_model_save_path=params.bestModelDir, log_path=params.evalDir, n_eval_episodes=n_eval_episodes,
    #     #                               callback_after_eval=CallbackAfterEval(), eval_freq=eval_freq, deterministic=True)
    #     # 达到最大指定的回合数时提前终止训练
    #     #callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=max_episodes, verbose=1)
    #     callback = [checkpoint_callback, my_callback]
    #     # 定义回调函数 ---end---

    # 训练或测试模型
    if params.Train != 0:
        print("加载模型，路径：" + params.loadModelDir)
        model.set_parameters(params.loadModelDir)
    if params.Train == 0:
        print("重头开始训练模型！ 模型保存路径：" + params.modelDir)
        print("日志保存路径：" + params.tensorboardDir)
        # model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    elif params.Train == 1:
        print("继续训练模型！ 模型保存路径：" + params.modelDir)
        print("日志保存路径：" + params.tensorboardDir)
        # model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False, progress_bar=True)
    else:
        print("测试模型，共测试 " + str(params.test_episodes) + " 回合！")
        obs = combatEnv.reset()
        win = 0
        draw = 0
        total_reward = 0
        episodes = 0
        start_time = time()
        for step in range(test_timesteps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = combatEnv.step(action)
            for env_idx in range(num_envs):
                if terminated[env_idx]:
                    episodes += 1
                    total_reward += info[env_idx]['episode']['r']
                    if role == "red" and info[env_idx]['result'] == 1 or role == "blue" and info[env_idx]['result'] == 2:
                        win += 1
                    if info[env_idx]['result'] == 3:
                        draw += 1

            if episodes >= params.test_episodes:
                break
        end_time = time()
        execution_time = end_time - start_time
        print("***********************************************************************")
        print("测试结束！共测试 " + str(episodes) + " 回合！胜利 " + str(win) + " 回合！平局" + str(draw) + "回合！")
        print("胜率：" + str(win / params.test_episodes))
        print("平均奖励：" + str(total_reward / params.test_episodes))
        print(f"Execution time: {execution_time/60:.2f} min ({execution_time:.2f} s)")


