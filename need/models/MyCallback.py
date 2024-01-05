from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import numpy as np
import configparser


class MyCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, policy_interval=0, save_dir="", best_save_dir="", monitor_dir="", load_dir="", eval_freq=10000, train_flag=0, verbose=0):
        super(MyCallback, self).__init__(verbose)

        self.policy_interval = policy_interval
        self.saveDir = save_dir
        self.bestSaveDir = best_save_dir
        self.monitor_dir = monitor_dir
        self.loadDir = load_dir
        self.eval_freq = eval_freq
        self.train_flag = train_flag
        self.best_mean_reward = -np.inf

        # 创建 ConfigParser 对象
        self.config = configparser.ConfigParser()
        self.config.add_section('parameters')


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """

        if self.train_flag == 1:
            # 读取配置文件
            # config = configparser.ConfigParser()
            # result = config.read(self.loadDir + 'parameters.ini')
            # if result:
            #     self.model.num_timesteps = int(config.get('parameters', 'num_timesteps'))   # 获取 num_timesteps
            # 加载回放缓冲区
            self.model.load_replay_buffer(self.loadDir + "replay_buffer.pkl")

        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # infos = self.locals["infos"][0]
        # self.logger.record("total_rewards", infos["reward"])
        # self.logger.record("potential_reward", infos["potential_reward"])

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.training_env.set_isEvaluate(True)

        if self.policy_interval > 0 and self.n_calls % self.policy_interval == 0:
            x, y = ts2xy(load_results(self.monitor_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.bestSaveDir + "best_models.zip")
                    self.model.policy.save(self.bestSaveDir + "best_policy.zip")  # 保存模型策略
                    self.model.save_replay_buffer(self.bestSaveDir + "replay_buffer.pkl")

            self.model.save(self.saveDir + "models.zip") #保存模型
            self.model.policy.save(self.saveDir + "policy.zip") #保存模型策略
            self.model.save_replay_buffer(self.saveDir + "replay_buffer.pkl")

            # 写入配置文件
            self.config.set('parameters', 'num_timesteps', str(self.model.num_timesteps))
            self.config.set('parameters', 'learning_rate', str(self.model.learning_rate))
            with open(self.saveDir + 'parameters.ini', 'w') as configfile:
                self.config.write(configfile)

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass