from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import Env


class MyCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, policy_interval=0, save_dir="", env=Env, eval_freq=10000, verbose=0):
        super(MyCallback, self).__init__(verbose)

        self.policy_interval = policy_interval
        self.saveDir = save_dir
        self.env = env
        self.eval_freq = eval_freq


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
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
            self.env.set_isEvaluate(True)

        if self.policy_interval > 0 and self.n_calls % self.policy_interval == 0:
            self.model.save(self.saveDir + "models.zip")

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