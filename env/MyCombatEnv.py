import gymnasium as gym
import numpy as np
import socket
import struct
import json

from models.params import StateDim, ActionDim
from env.obsToState import ObsToState
from utils.RongAoUtils import RongAoUtils


# 定义环境类
class MyCombatEnv(gym.Env):
    #metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self, role='red', role_id='1001', render_mode=None):
        super().__init__()
        # 使用连续动作空间
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=ActionDim, dtype=np.float32)
        # 使用连续状态空间
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=StateDim, dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # 创建仿真链接，初始化环境交互类信息
        # 端口连接
        self.Red_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.Red_client.settimeout(5)  # 设置超时时间为 5 秒
        self.Red_client.connect(('127.0.0.1', 8868))

        self.Red_identify_dict = {
            "msg_type": "identify",
            "msg_info": {
                "identify": "red"}}
        self.manu_ctrl = {
            "msg_info": "驾驶操控,1001,2,0,Delay,Force,0|0`0`0.8`0",
            "msg_type": "manu_ctrl",
            "done": "0"}  # 是否结束战斗并附带原因[0-未结束 1-本方胜 2-敌方胜[本机摔或高度过低] 3-时间到]
        # self.manu_ctrl_launch = {
        #     "msg_info": "发射, 1001, 2, 0, Delay, Force, 0 | 1002",
        #     "msg_type": "manu_ctrl"}

        self.Red_identify_dict["msg_info"]["identify"] = role
        self.role = role
        self.id = role_id
        self.rcv_msg = bytes()  # 缓存接收到的报文
        self.cur_manu_ctrl = self.manu_ctrl
        self.CurTime = 0
        self.PreTime = 0
        self.msg_type = None
        self.pre_msg_type = None
        self.nowInfo = None
        self.preInfo = None
        self.firstFrameInfo = None
        self.secondFrameInfo = None
        self.thirdFrameInfo = None
        self.secondFrameTime = 0
        self.latest_observation = np.zeros(StateDim, dtype=np.float32)
        self.latest_reward = 0
        self.potential_reward = 0
        self.isDone = 0
        self.obsToState = ObsToState()
        # 初始化环境交互类信息
        self.sendMsg(self.Red_identify_dict)

    def step(self, action):
        """
        环境更新,31代表机动，32代表打击
        """
        observation = np.zeros(StateDim, dtype=np.float32)
        reward = 0
        terminated = False
        truncated = False
        info = {}
        # 发送动作指令
        self.cur_manu_ctrl = RongAoUtils.moveRL(self.cur_manu_ctrl, self.id, action[0], action[1], action[2], action[3])
        self.cur_manu_ctrl["done"] = self.isDone  # 结束标志
        self.sendMsg(self.cur_manu_ctrl)
        # 更新数据信息
        self.secondFrameTime = self.CurTime
        # 接收环境信息
        self.receiveMsg()
        self.firstFrameInfo = self.secondFrameInfo
        self.secondFrameInfo = self.thirdFrameInfo
        self.thirdFrameInfo = self.nowInfo
        if self.msg_type == "result":
            print("已收到对局结束指令!")
            terminated = True
        elif self.role == "red" and self.msg_type == "red_out_put" or self.role == "blue" and self.msg_type == "blue_out_put":
            self.getStateReward()
            observation = self.latest_observation
            reward = self.latest_reward
            if self.isDone != 0:
                # 发送动作指令
                self.cur_manu_ctrl = RongAoUtils.moveRL(self.cur_manu_ctrl, self.id, action[0], action[1], action[2], action[3])
                self.cur_manu_ctrl["done"] = self.isDone  # 结束标志
                self.sendMsg(self.cur_manu_ctrl)
                terminated = True
                # if self.isDone == 3:
                #     truncated = True
                # else:
                #     terminated = True
        else:
            print("接收到不明信息!")
            truncated = True
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 重置环境
        self.rcv_msg = bytes()
        self.cur_manu_ctrl = self.manu_ctrl
        self.CurTime = 0
        self.PreTime = 0
        self.msg_type = None
        self.pre_msg_type = None
        self.nowInfo = None
        self.preInfo = None
        self.firstFrameInfo = None
        self.secondFrameInfo = None
        self.thirdFrameInfo = None
        self.secondFrameTime = 0
        self.latest_observation = np.zeros(StateDim, dtype=np.float32)
        self.latest_reward = 0
        self.potential_reward = 0
        self.isDone = 0
        observation = np.zeros(StateDim, dtype=np.float32)
        info = {}
        # 获得第一帧信息
        while self.preInfo is None:
            self.receiveMsg()
            self.preInfo = self.nowInfo
        self.sendMsg(self.cur_manu_ctrl)
        self.firstFrameInfo = self.nowInfo
        # 获得第二帧信息
        self.preInfo = None
        while self.preInfo is None:
            self.receiveMsg()
            self.preInfo = self.nowInfo
        self.sendMsg(self.cur_manu_ctrl)
        self.secondFrameInfo = self.nowInfo
        self.secondFrameTime = self.CurTime
        # 获得第三帧信息
        self.receiveMsg()
        self.thirdFrameInfo = self.nowInfo
        if self.role == "red" and self.msg_type == "red_out_put" or self.role == "blue" and self.msg_type == "blue_out_put":
            self.getStateReward()
            observation = self.latest_observation
        info = self._get_info()
        return observation, info

    def sendMsg(self, msg):
        msg_len = len(json.dumps(msg).encode("utf-8"))
        msg_len_data = struct.pack('i', msg_len)
        self.Red_client.send(msg_len_data)
        self.Red_client.send(json.dumps(msg).encode("utf-8"))

    def getStateReward(self):
        self.latest_observation, self.latest_reward, self.isDone = self.obsToState.getStateRewardDone(
            self.firstFrameInfo, self.secondFrameInfo, self.thirdFrameInfo, self.secondFrameTime)
        self.preInfo = self.nowInfo
        self.PreTime = self.CurTime
        self.pre_msg_type = self.msg_type

    def receiveMsg(self):
        try:
            cur_msg = self.Red_client.recv(1024 * 10)
            self.rcv_msg = self.rcv_msg + cur_msg
            total_len = len(self.rcv_msg)
            while total_len > 4:
                msg_len = int.from_bytes(self.rcv_msg[0:4], byteorder="little", signed=False)
                if (msg_len + 4 <= total_len):
                    # 一帧数据
                    one_frame_msg = self.rcv_msg[4:4 + msg_len]
                    # 去除一帧后的剩余数据
                    self.rcv_msg = self.rcv_msg[4 + msg_len: total_len]
                    total_len = len(self.rcv_msg)
                    Red_json_str_recv = json.loads(one_frame_msg)
                    self.msg_type = Red_json_str_recv["msg_type"]
                    self.CurTime = Red_json_str_recv["msg_time"]
                    self.nowInfo = Red_json_str_recv["msg_info"]
                else:
                    break
        except socket.timeout:
            # 处理超时情况
            print("接收数据超时")
            self.msg_type = self.pre_msg_type
            self.CurTime = self.PreTime
            self.nowInfo = self.preInfo

    def close(self):
        self.Red_client.close()

    def _get_info(self):
        return {
            # "reward": self.latest_reward,
            # "done": self.isDone != 0,
            # "potential_reward": self.potential_reward
        }



#check env
# from stable_baselines3.common.env_checker import check_env
# env = MyCombatEnv('red', '1001')
# check_env(env)

