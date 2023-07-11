import math
import numpy as np
from utils.RongAoUtils import RongAoUtils
from models.params import StateDim

class ObsToState:
    def __init__(self):
        self.isDone = 0  #是否结束战斗并附带原因[0-未结束 1-本方胜 2-敌方胜[本机摔或高度过低] 3-时间到]
        self.CurTime = 0
        self.CurTotalReward = 0
        self.CurPotentialReward = 0
        self.CurDistance = 0
        self.CurAttackAngle = 0
        self.CurEscapeAngle = 0
        self.WEZ_Min = 200
        self.WEZ_Max = 1000
        self.Distance_Max = 10000
        self.ALT_Min = 1000
        self.ALT_Max = 10000
        self.TotalReward = 0

    # 获得状态奖励结束标志
    def getStateRewardDone(self, first_info, second_info, third_info, time):
        """
        根据观测信息（3帧），返回状态，奖励函数，是否结束
        """
        self.CurTime = time
        self.isDone = 0
        #解析第1帧数据
        pre_self_track, pre_DetectedInfo = self.parseOneFrameData(first_info)
        #解析第2帧数据
        cur_self_track, cur_DetectedInfo = self.parseOneFrameData(second_info)
        #解析第3帧数据
        next_self_track, next_DetectedInfo = self.parseOneFrameData(third_info)
        #提取第1帧观测数据
        pre_state = self.getOneFrameState(pre_self_track, pre_DetectedInfo)
        #提取第2帧观测数据
        cur_state = self.getOneFrameState(cur_self_track, cur_DetectedInfo)
        #提取第3帧观测数据
        next_state = self.getOneFrameState(next_self_track, next_DetectedInfo)
        #状态向量
        STATE = np.concatenate((pre_state, cur_state, next_state)).reshape(StateDim).astype(np.float32)

        cur_sparse_reward, self.CurPotentialReward = self.getReward(cur_self_track, cur_DetectedInfo, True)
        next_sparse_reward, next_potential_reward = self.getReward(next_self_track, next_DetectedInfo, False)

        # reward function
        self.CurTotalReward = cur_sparse_reward + next_potential_reward - self.CurPotentialReward # + 0.001*math.tanh(-self.CurTime / 300)
        # termination probability
        terminate_prob = self.CurPotentialReward
        if terminate_prob < 0:
            terminate_prob = 0

        self.is_termination(terminate_prob, cur_self_track)

        return STATE, self.CurTotalReward, self.isDone

    def is_termination(self, terminate_prob, cur_self_info):
        #判断是否结束
        if terminate_prob > 0.7:
            self.isDone = 1
            terminal_reward = 50 + math.exp(-2*self.CurTime/300)
            self.CurTotalReward += terminal_reward
            self.TotalReward += self.CurTotalReward
            print("--------------------------------------------------------目标达成！---------------------------------------------------------------------")
            print("本局时间: ", self.CurTime, ", TotalReward=", self.TotalReward)
            print(", 高度:", cur_self_info['Altitude'], ", distance=", self.CurDistance,
                  ", attackAngle=", self.CurAttackAngle*180/math.pi, ", escapeAngle=", self.CurEscapeAngle*180/math.pi)

        elif self.CurDistance > self.Distance_Max:
            self.isDone = 2
            terminal_reward = -50 + math.tanh(-math.fabs(self.CurAttackAngle) / math.pi)
            self.CurTotalReward += terminal_reward
            self.TotalReward += self.CurTotalReward
            print("超出最大距离范围，结束  时间: ", self.CurTime, ", TotalReward=", self.TotalReward, ", CurAttackAngle=",
                  self.CurAttackAngle * 180 / math.pi)

        elif cur_self_info['Altitude'] > self.ALT_Max or cur_self_info['Altitude'] < self.ALT_Min:
            self.isDone = 2
            terminal_reward = -40 + math.tanh((self.WEZ_Min - self.CurDistance) / self.Distance_Max)
            self.CurTotalReward += terminal_reward
            self.TotalReward += self.CurTotalReward
            print("高度越界，结束  时间: ", self.CurTime, ", TotalReward=", self.TotalReward, ", distance=", self.CurDistance)

        elif self.CurTime > 299:
            self.isDone = 3
            terminal_reward = -30 + math.tanh((self.WEZ_Min - self.CurDistance) / self.Distance_Max)
            self.CurTotalReward += terminal_reward
            self.TotalReward += self.CurTotalReward
            print("---超时---，", "TotalReward=", self.TotalReward, ", distance=", self.CurDistance)

        if self.isDone != 0:
            self.TotalReward = 0

    # 计算奖励
    def getReward(self, self_info, dete_info, curFlag=False):
        # 计算两机距离
        distance = RongAoUtils.getDistance3D(dete_info)
        # 计算攻击角[-pi, pi]、逃逸角[-pi, pi]
        distanceVector3D = RongAoUtils.getDistanceVector3D(dete_info)
        selfPlaneSpeedVector = RongAoUtils.getSpeedVector3D(self_info)
        detePlaneSpeedVector = RongAoUtils.getSpeedVector3D(dete_info)
        attackAngle = math.acos(
            (selfPlaneSpeedVector[0] * distanceVector3D[0] + selfPlaneSpeedVector[1] * distanceVector3D[1] +
             selfPlaneSpeedVector[2] * distanceVector3D[2]) / distance)
        escapeAngle = math.acos(
            (detePlaneSpeedVector[0] * distanceVector3D[0] + detePlaneSpeedVector[1] * distanceVector3D[1] +
             detePlaneSpeedVector[2] * distanceVector3D[2]) / distance)

        # sparse reward function
        sparse_reward = 0
        # 敌机在我机范围内
        if math.fabs(attackAngle) <= math.pi / 3 and math.fabs(escapeAngle) <= math.pi / 3 and \
                (distance >= self.WEZ_Min and distance <= self.WEZ_Max):
            sparse_reward = 1
        # 我机在敌机范围内
        if math.fabs(attackAngle) >= 2 * math.pi / 3 and math.fabs(escapeAngle) >= 2 * math.pi / 3 and \
                (distance >= self.WEZ_Min and distance <= self.WEZ_Max):
            sparse_reward = -1

        # potential function
        potential_reward = (1 - (math.fabs(attackAngle) / math.pi + math.fabs(escapeAngle) / math.pi)) * \
                           math.exp(-0.0002 * math.fabs(distance - self.WEZ_Min))

        if curFlag is True:
            self.CurDistance = distance
            self.CurAttackAngle = attackAngle
            self.CurEscapeAngle = escapeAngle

        return sparse_reward, potential_reward


    # 获得一帧的状态
    def getOneFrameState(self, self_track, detectedInfo):
        # 提取1帧观测数据
        # 经纬度差[-180,180]、[-90,90]
        lon_diff = (detectedInfo['Longitude'] - self_track['Longitude']) / 180.0
        lat_diff = (detectedInfo['Latitude'] - self_track['Latitude']) / 90.0
        # 高度
        self_alt = self_track['Altitude'] / self.ALT_Max
        dete_alt = detectedInfo['Altitude'] / self.ALT_Max
        # 速度分量
        self_v_n = self_track['V_N'] / 600.0
        self_v_e = self_track['V_E'] / 600.0
        self_v_d = self_track['V_D'] / 600.0
        # 速度差[0, 600]
        self_v_real = RongAoUtils.getSpeed(self_track)
        dete_v_real = RongAoUtils.getSpeed(detectedInfo)
        v_diff = (dete_v_real - self_v_real) / 600.0
        # 偏航[0, 2*pi]、俯仰[-pi/2, pi/2]、滚转[-pi, pi]
        self_heading = self_track['Heading'] / (2 * math.pi)
        self_pitch = self_track['Pitch'] / (math.pi / 2)
        self_roll = self_track['Roll'] / math.pi
        dete_heading = detectedInfo['Heading'] / (2 * math.pi)
        dete_pitch = detectedInfo['Pitch'] / (math.pi / 2)
        dete_roll = detectedInfo['Roll'] / math.pi
        # 状态向量
        one_state = np.array([lon_diff, lat_diff, self_alt, dete_alt, self_v_n, self_v_e, self_v_d, v_diff,
                              self_heading, self_pitch, self_roll, dete_heading, dete_pitch, dete_roll],
                             dtype=np.float32)

        return one_state

    # 解析一帧数据
    def parseOneFrameData(self, observation):
        cur_DetectedInfo = {}
        cur_WeaponSystem = {}
        cur_self_track = {}
        cur_SFC = {}
        cur_MissileTrack = {}
        for data in observation:
            # 探测信息
            if data["data_tp"] == "DetectedInfo":
                if len(data["data_info"][0]['DetectedTargets']) > 0:
                    cur_DetectedInfo = data["data_info"][0]['DetectedTargets'][0]
            # 武器信息
            elif data["data_tp"] == "WeaponSystem":
                cur_WeaponSystem = data["data_info"][0]
            # 自身航迹信息
            elif data["data_tp"] == "track":
                cur_self_track = data["data_info"][0]
            # 火控信息
            elif data["data_tp"] == "SFC":
                cur_SFC = data["data_info"]
            # 导弹信息
            elif data["data_tp"] == "MissileTrack":
                cur_MissileTrack = data["data_info"]

        return cur_self_track, cur_DetectedInfo






