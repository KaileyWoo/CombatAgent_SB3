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
        my_env = gym.make(env_id, role=role, role_id=role_id, env_id=rank)
        return my_env
    return _init

def main(role='red', role_id='1001'):
    # ��������ļ�·��
    params = Params(role=role)

    # �����໷��
    print("������ {} �黷��".format(num_envs))
    envs = [make_env(env_id='MyCombatEnv-v0', rank=i, role=role, role_id=role_id) for i in range(num_envs)]
    combatEnv = SubprocVecEnv(envs)
    combatEnv = VecMonitor(combatEnv, params.monitorDir)

    # ����ģ��
    model = SAC('MlpPolicy', combatEnv, verbose=0, device='cuda', tensorboard_log=params.tensorboardDir,
                batch_size=batch_size, learning_rate=learning_rate, buffer_size=buffer_size)

    # ����ص����� ---begin---
    # ����ģ�͵��м�״̬
    checkpoint_callback = CheckpointCallback(save_freq=Checkpoint_interval, save_path=params.CheckpointDir, name_prefix='sac_model')
                                              #save_replay_buffer=True, save_vecnormalize=True)
    # �Զ���ص�������1.��ʱ������ԣ�2.����ѵ�������н���ģ�������������û���������ģʽ��־λ��
    my_callback = MyCallback(policy_interval=policy_interval, save_dir=params.modelDir, env=combatEnv, eval_freq=0)
    # ��ѵ�������н���ģ����������
    # eval_callback = EvalCallback(combatEnv, best_model_save_path=params.bestModelDir, log_path=params.evalDir, n_eval_episodes=n_eval_episodes,
    #                               callback_after_eval=CallbackAfterEval(), eval_freq=eval_freq, deterministic=True)
    # �ﵽ���ָ���Ļغ���ʱ��ǰ��ֹѵ��
    #callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=max_episodes, verbose=1)
    callback = [checkpoint_callback, my_callback]
    # ����ص����� ---end---

    # ѵ�������ģ��
    if params.Train != 0:
        print("����ģ�ͣ�·����" + params.loadModelDir)
        model.set_parameters(params.loadModelDir)
    if params.Train == 0:
        print("��ͷ��ʼѵ��ģ�ͣ� ģ�ͱ���·����" + params.modelDir)
        print("��־����·����" + params.tensorboardDir)
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    elif params.Train == 1:
        print("����ѵ��ģ�ͣ� ģ�ͱ���·����" + params.modelDir)
        print("��־����·����" + params.tensorboardDir)
        model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False, progress_bar=True)
    else:
        print("����ģ�ͣ������� " + str(test_episodes) + " �غϣ�")
        obs = combatEnv.reset()
        win = 0
        draw = 0
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
                    if role == "red" and info[env_idx]['result'] == 1 or role == "blue" and info[env_idx]['result'] == 2:
                        win += 1
                    if info[env_idx]['result'] == 3:
                        draw += 1

            if episodes >= test_episodes:
                break
        end_time = time.time()
        execution_time = end_time - start_time
        print("***********************************************************************")
        print("���Խ����������� " + str(episodes) + " �غϣ�ʤ�� " + str(win) + " �غϣ�ƽ��" + str(draw) + "�غϣ�")
        print("ʤ�ʣ�" + str(win / test_episodes))
        print("ƽ��������" + str(total_reward / test_episodes))
        print(f"Execution time: {execution_time/60:.2f} min ({execution_time:.2f} s)")



