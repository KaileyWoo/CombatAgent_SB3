import gymnasium as gym
import numpy as np
import socket
import struct
import json

from models.params import StateDim, ActionDim, n_eval_episodes
from env.obsToState import ObsToState
from utils.RongAoUtils import RongAoUtils

# ���廷����
class MyCombatEnv(gym.Env):
    def __init__(self, role='red', role_id='1001', env_id=0, render_mode=None):
        super().__init__()
        self.role = role
        self.id = role_id
        self.env_id = env_id
        self.detectedId = '1002'

        # ʹ�����������ռ�
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=ActionDim, dtype=np.float32)
        # ʹ������״̬�ռ�
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=StateDim, dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # �����������ӣ���ʼ��������������Ϣ
        # �˿�����
        self.Red_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.Red_client.connect(('127.0.0.1', 8868+env_id))

        self.Red_identify_dict = {
            "msg_type": "identify",
            "msg_info": {
                "identify": "red"}}
        self.manu_ctrl_launch = self.generate_manu_ctrl_launch_msg()
        self.cur_manu_ctrl = self.generate_manu_ctrl_msg()
        self.Red_identify_dict["msg_info"]["identify"] = self.role

        self.rcv_msg = bytes()  # ������յ��ı���
        self.CurTime = 0
        self.PreTime = 0
        self.msg_type = None
        self.pre_msg_type = None
        self.nowInfo = None
        self.preInfo = None
        self.firstFrameInfo = None
        self.secondFrameInfo = None
        self.thirdFrameInfo = None
        self.firstFrameTime = 0
        self.secondFrameTime = 0
        self.latest_observation = np.zeros(StateDim, dtype=np.float32)
        self.latest_reward = 0
        self.potential_reward = 0
        self.obsToState = ObsToState(self.env_id, 8868+env_id)
        self.isEvaluate = False
        self.num_eval = 0
        self.isDone = 0
        self.result = 0
        # ��ʼ��������������Ϣ
        self.sendMsg(self.Red_identify_dict)


    def generate_manu_ctrl_msg(self):
        msg_info = [f"��ʻ�ٿ�,{self.id},2,0,Delay,Force,0|0`0`0.6`0"]
        return {
            "msg_info": msg_info,
            "msg_type": "manu_ctrl",
            "done": 0
        }


    def generate_manu_ctrl_launch_msg(self):
        msg_info = [f"��ʻ�ٿ�,{self.id},2,0,Delay,Force,0|0`0`0.6`0", f"����,{self.id},2,0,Delay,Force,0|{self.detectedId}"]
        return {
            "msg_info": msg_info,
            "msg_type": "manu_ctrl",
            "done": 0
        }


    def step(self, action):
        """
        ��������,31���������32������
        """
        observation = np.zeros(StateDim, dtype=np.float32)
        reward = 0
        terminated = False
        truncated = False
        # ���Ͷ���ָ��
        self.cur_manu_ctrl = RongAoUtils.moveRL(self.cur_manu_ctrl, self.id, action[0], action[1], action[2], action[3])
        self.cur_manu_ctrl["done"] = self.isDone  # ������־
        self.sendMsg(self.cur_manu_ctrl)

        # ����������Ϣ,���ջ�����Ϣ
        self.receiveMsg()
        self.firstFrameInfo = self.secondFrameInfo
        self.secondFrameInfo = self.nowInfo
        if self.msg_type == "result":
            self.result = self.nowInfo["result"]
            reward = self.obsToState.proTerminal(self.result, self.role)
            observation = self.latest_observation
            terminated = True
        elif self.role == "red" and self.msg_type == "red_out_put" or self.role == "blue" and self.msg_type == "blue_out_put":
            self.getStateReward()
            observation = self.latest_observation
            reward = self.latest_reward
            if self.isDone != 0:
                self.cur_manu_ctrl["done"] = self.isDone  # ������־
                self.sendMsg(self.cur_manu_ctrl)
                # self.sendMsg(self.manu_ctrl_launch)
                # terminated = True
        else:
            print("���յ�������Ϣ!")
            truncated = True

        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # ��֤����
        # if self.isEvaluate and self.isDone == 0:
        #     self.cur_manu_ctrl["done"] = 2
        #     self.sendMsg(self.cur_manu_ctrl)
        #     self.obsToState.resetEpisode()
        # if self.num_eval >= n_eval_episodes:
        #     self.isEvaluate = False
        #     self.num_eval = 0
        #     print("������������������������������������ end����������������������������")
        # if self.isEvaluate and self.num_eval < n_eval_episodes:
        #     if self.num_eval == 0:
        #         print("����������������������������������ʼ begin, ��", n_eval_episodes,"�֡���������������������������")
        #     self.num_eval += 1

        # ���û���
        self.cur_manu_ctrl = self.generate_manu_ctrl_msg()
        self.rcv_msg = bytes()
        self.CurTime = 0
        self.PreTime = 0
        self.msg_type = None
        self.pre_msg_type = None
        self.nowInfo = None
        self.preInfo = None
        self.firstFrameInfo = None
        self.secondFrameInfo = None
        self.thirdFrameInfo = None
        self.firstFrameTime = 0
        self.secondFrameTime = 0
        self.latest_observation = np.zeros(StateDim, dtype=np.float32)
        self.latest_reward = 0
        self.potential_reward = 0
        self.isDone = 0
        self.result = 0
        observation = np.zeros(StateDim, dtype=np.float32)
        info = {}
        self.towFrames()
        if self.role == "red" and self.msg_type == "red_out_put" or self.role == "blue" and self.msg_type == "blue_out_put":
            if len(self.nowInfo) != 0:
                self.latest_observation = self.obsToState.getState(self.firstFrameInfo, self.secondFrameInfo, self.CurTime)
                self.preInfo = self.nowInfo
                self.PreTime = self.CurTime
                self.pre_msg_type = self.msg_type
                observation = self.latest_observation
        info = self._get_info()
        return observation, info

    def towFrames(self):
        # ��õ�һ֡��Ϣ
        while self.preInfo is None:
            self.receiveMsg()
            self.preInfo = self.nowInfo
        self.sendMsg(self.cur_manu_ctrl)
        self.firstFrameInfo = self.nowInfo
        # ��õڶ�֡��Ϣ
        self.receiveMsg()
        self.secondFrameInfo = self.nowInfo

    def sendMsg(self, msg):
        msg_len = len(json.dumps(msg).encode("utf-8"))
        msg_len_data = struct.pack('i', msg_len)
        self.Red_client.send(msg_len_data)
        self.Red_client.send(json.dumps(msg).encode("utf-8"))

    def getStateReward(self):
        if len(self.nowInfo) != 0:
            self.latest_observation, self.latest_reward, self.isDone = self.obsToState.getStateRewardDoneForTwoFrames(
                self.firstFrameInfo, self.secondFrameInfo, self.CurTime)
            self.preInfo = self.nowInfo
            self.PreTime = self.CurTime
            self.pre_msg_type = self.msg_type
        else:
            self.isDone = 2

    def receiveMsg(self):
        cur_msg = self.Red_client.recv(1024 * 10)
        self.rcv_msg = self.rcv_msg + cur_msg
        total_len = len(self.rcv_msg)
        while total_len > 4:
            msg_len = int.from_bytes(self.rcv_msg[0:4], byteorder="little", signed=False)
            if (msg_len + 4 <= total_len):
                # һ֡����
                one_frame_msg = self.rcv_msg[4:4 + msg_len]
                # ȥ��һ֡���ʣ������
                self.rcv_msg = self.rcv_msg[4 + msg_len: total_len]
                total_len = len(self.rcv_msg)
                Red_json_str_recv = json.loads(one_frame_msg)
                self.msg_type = Red_json_str_recv["msg_type"]
                self.CurTime = Red_json_str_recv["msg_time"]
                self.nowInfo = Red_json_str_recv["msg_info"]
            else:
                break

    def close(self):
        self.Red_client.close()

    def _get_info(self):
        return {
            "done": self.isDone,
            "result": self.result,
        }

    def set_isEvaluate(self, flag=False):
        self.isEvaluate = flag



#check env
# from stable_baselines3.common.env_checker import check_env
# env = MyCombatEnv('red', '1001')
# check_env(env)

