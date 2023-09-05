from stable_baselines3.common.callbacks import BaseCallback
#import gymnasium as gym


class CallbackAfterEval(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self,verbose=0):
        super(CallbackAfterEval, self).__init__(verbose)


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

        # 获取模型的优化器
        optimizer = self.model.policy.optimizer
        # 定义学习率衰减函数，这里使用指数衰减
        decay_factor = 0.9  # 衰减因子，可根据需求调整
        current_lr = optimizer.param_groups[0]['lr']  # 当前学习率
        new_lr = current_lr * decay_factor  # 计算新学习率
        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        i = 0
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