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
        self.CurSelfTrack = None
        self.CurDetectedInfo = None
        self.PreSelfTrack = None
        self.PreDetectedInfo = None
        self.WEZ_Min = 200
        self.WEZ_Max = 1000
        self.Distance_Max = 10000
        self.ALT_Min = 1000
        self.ALT_Max = 10000
        self.TIME_Max = 300
        self.TotalReward = 0
        self.episode = 1


    def getState(self, first_info, second_info, time):
        self.CurTime = time
        # 解析第1帧数据
        self.PreSelfTrack, self.PreDetectedInfo = self.parseOneFrameData(first_info)
        # 解析第2帧数据
        self.CurSelfTrack, self.CurDetectedInfo = self.parseOneFrameData(second_info)
        # 状态向量
        first_state = self.getStateVector(self.PreSelfTrack, self.PreDetectedInfo)
        second_stata = self.getStateVector(self.CurSelfTrack, self.CurDetectedInfo)
        STATE = np.concatenate((first_state, second_stata)).reshape(StateDim).astype(np.float32)
        return STATE


    def getStateRewardDoneForTwoFrames(self, first_info, second_info, time):
        """
        根据观测信息（2帧），返回状态，奖励函数，是否结束
        """
        self.isDone = 0

        STATE = self.getState(first_info, second_info, time)

        # 计算奖励
        self.CurTotalReward = self.getStepReward(self.PreSelfTrack, self.CurSelfTrack, self.PreDetectedInfo, self.CurDetectedInfo)
        self.is_done(self.CurSelfTrack)

        return STATE, self.CurTotalReward, self.isDone


    # 判断是否结束
    def is_done(self, self_track):
        if self.CurDistance > self.Distance_Max:
            Reward = -50 + math.tanh(-math.fabs(self.CurAttackAngle) / math.pi)
            self.CurTotalReward = Reward
            self.TotalReward += self.CurTotalReward
            self.isDone = 2
            str_ = f"第{self.episode}局结束，***距离***越界，时间：{self.CurTime:.1f}s，奖励：{self.TotalReward:.6f}，距离：{self.CurDistance:.2f}m，" \
                   f"高度：{self_track['Altitude']:.2f}m，攻击角：{self.CurAttackAngle * 180 / math.pi:.2f}，逃逸角：{self.CurEscapeAngle * 180 / math.pi:.2f}"
            print(str_)
        elif self_track['Altitude'] > self.ALT_Max or self_track['Altitude'] < self.ALT_Min:
            Reward = -50 + math.tanh((self.WEZ_Max - self.CurDistance) / self.Distance_Max)
            self.CurTotalReward = Reward
            self.TotalReward += self.CurTotalReward
            self.isDone = 2
            str_ = f"第{self.episode}局结束，+++高度+++越界，时间：{self.CurTime:.1f}s，奖励：{self.TotalReward:.6f}，距离：{self.CurDistance:.2f}m，" \
                   f"高度：{self_track['Altitude']:.2f}m，攻击角：{self.CurAttackAngle * 180 / math.pi:.2f}，逃逸角：{self.CurEscapeAngle * 180 / math.pi:.2f}"
            print(str_)
        elif self.CurTime >= self.TIME_Max:
            Reward = -50 + 0.8 * math.tanh((self.WEZ_Max - self.CurDistance) / self.Distance_Max) + 0.2 * math.tanh(-math.fabs(self.CurAttackAngle) / math.pi)
            self.CurTotalReward = Reward
            self.TotalReward += self.CurTotalReward
            self.isDone = 3
            str_ = f"第{self.episode}局结束，---时间---越界，时间：{self.CurTime:.1f}s，奖励：{self.TotalReward:.6f}，距离：{self.CurDistance:.2f}m，" \
                   f"高度：{self_track['Altitude']:.2f}m，攻击角：{self.CurAttackAngle * 180 / math.pi:.2f}，逃逸角：{self.CurEscapeAngle * 180 / math.pi:.2f}"
            print(str_)
        elif math.fabs(self.CurAttackAngle) < math.pi / 3 and self.CurDistance < self.WEZ_Max:
            Reward = 50 + 0.8*math.exp(-self.CurTime / self.TIME_Max) + 0.2*math.exp(-(math.fabs(self.CurEscapeAngle))/math.pi)
            self.CurTotalReward = Reward
            self.TotalReward += self.CurTotalReward
            self.isDone = 1
            print("--------------------------------------------------------目标达成！---------------------------------------------------------------------")
            str_ = f"第{self.episode}局结束，时间: {self.CurTime:.1f}s，奖励：{self.TotalReward:.6f}，距离：{self.CurDistance:.2f}m，" \
                   f"高度：{self_track['Altitude']:.2f}m，攻击角：{self.CurAttackAngle * 180 / math.pi:.2f}，逃逸角：{self.CurEscapeAngle * 180 / math.pi:.2f}"
            print(str_)

        if self.isDone != 0:
            self.TotalReward = 0
            self.episode += 1
        else:
            self.TotalReward += self.CurTotalReward


    # 计算单步奖励
    def getStepReward(self, first_self_track, second_self_track, first_DetectedInfo, second_DetectedInfo):

        pre_distance, pre_self_attackAngle, pre_target_escapeAngle = self.calDisAttakEscape(first_self_track, first_DetectedInfo)
        distance, self_attackAngle, target_escapeAngle = self.calDisAttakEscape(second_self_track, second_DetectedInfo)
        self.CurDistance = distance
        self.CurAttackAngle = self_attackAngle
        self.CurEscapeAngle = target_escapeAngle

        # 距离奖励
        pre_distance -= self.WEZ_Max
        distance -= self.WEZ_Max
        dis_reward = math.tanh((pre_distance - distance) / 10)
        # 攻击角奖励
        attack_reward = math.tanh((pre_self_attackAngle - self_attackAngle) * 100)
        # 逃逸角奖励
        escape_reward = math.tanh((pre_target_escapeAngle - target_escapeAngle) * 100)

        Reward = 0.4 * dis_reward + 0.4 * attack_reward + 0.1 * escape_reward + 0.1 * math.tanh(-self.CurTime / self.TIME_Max)
        Reward = min(1, max(-1, Reward))
        Reward *= 0.01

        return Reward


    # 获得状态向量
    def getStateVector(self, self_info, dete_info):
        distance, self_attackAngle, target_escapeAngle = self.calDisAttakEscape(self_info, dete_info)
        # 高度
        self_alt = self_info['Altitude'] / self.ALT_Max
        dete_alt = dete_info['Altitude'] / self.ALT_Max
        # alt_diff = (self_track['Altitude'] - detectedInfo['Altitude']) / self.ALT_Max
        # 速度差[0, 600]
        self_v_real = RongAoUtils.getSpeed(self_info) / 600.0
        dete_v_real = RongAoUtils.getSpeed(dete_info) / 600.0
        #v_diff = (self_v_real - dete_v_real)
        # 速度分量
        self_v_n = self_info['V_N'] / 600.0
        self_v_e = self_info['V_E'] / 600.0
        self_v_d = self_info['V_D'] / 600.0
        # 加速度分量
        self_a_x = self_info['accelerations_x'] / 100.0
        self_a_y = self_info['accelerations_y'] / 100.0
        self_a_z = self_info['accelerations_z'] / 100.0
        # 偏航[0, 2*pi]、俯仰[-pi/2, pi/2]、滚转[-pi, pi]
        self_heading = self_info['Heading'] / (2 * math.pi)
        self_pitch = self_info['Pitch'] / (math.pi / 2)
        self_roll = self_info['Roll'] / math.pi
        dete_heading = dete_info['Heading'] / (2 * math.pi)
        dete_pitch = dete_info['Pitch'] / (math.pi / 2)
        dete_roll = dete_info['Roll'] / math.pi
        # 其他参数
        self_alpha = self_info['alpha'] / (2 * math.pi)    # 迎角/攻角
        self_beta = self_info['beta'] / (2 * math.pi)      # 侧滑角/滑移角
        self_p = self_info['p'] / math.pi  # 滚转角速度(弧度每秒)
        self_q = self_info['q'] / math.pi  # 俯仰角速度(弧度每秒)
        self_r = self_info['r'] / math.pi  # 侧滑角速度(弧度每秒)

        # 状态向量
        state = np.array([distance/self.Distance_Max, self_attackAngle/math.pi, target_escapeAngle/math.pi, self_alt, dete_alt,
                          self_v_real, dete_v_real, self_v_n, self_v_e, self_v_d, self_a_x, self_a_y, self_a_z,
                          self_heading, self_pitch, self_roll, dete_heading, dete_pitch, dete_roll,
                          self_alpha, self_beta, self_p, self_q, self_r],dtype=np.float32)

        return state


    # 获得状态奖励结束标志
    def getStateRewardDone(self, first_info, second_info, third_info, time):
        """
        根据观测信息（3帧），返回状态，奖励函数，是否结束
        """
        self.CurTime = time
        self.isDone = 0

        # 解析第1帧数据
        pre_self_track, pre_DetectedInfo = self.parseOneFrameData(first_info)
        # 解析第2帧数据
        cur_self_track, cur_DetectedInfo = self.parseOneFrameData(second_info)
        # 解析第3帧数据
        next_self_track, next_DetectedInfo = self.parseOneFrameData(third_info)
        # 提取第1帧观测数据
        pre_state = self.getOneFrameState(pre_self_track, pre_DetectedInfo)
        # 提取第2帧观测数据
        cur_state = self.getOneFrameState(cur_self_track, cur_DetectedInfo)
        # 提取第3帧观测数据
        next_state = self.getOneFrameState(next_self_track, next_DetectedInfo)
        # 状态向量
        STATE = np.concatenate((pre_state, cur_state, next_state)).reshape(StateDim).astype(np.float32)

        # cur_sparse_reward, self.CurPotentialReward = self.getReward(cur_self_track, cur_DetectedInfo, True)
        # next_sparse_reward, next_potential_reward = self.getReward(next_self_track, next_DetectedInfo, False)
        self.CurPotentialReward, terminate_prob = self.getReward(cur_self_track, cur_DetectedInfo, True)
        next_potential_reward, next_terminate_prob = self.getReward(next_self_track, next_DetectedInfo, False)

        # reward function
        self.CurTotalReward = next_potential_reward - self.CurPotentialReward  # + 0.01*math.tanh(-self.CurTime / self.TIME_Max)

        self.is_termination(terminate_prob, cur_self_track)

        return STATE, self.CurTotalReward, self.isDone


    def is_termination(self, terminate_prob, cur_self_info):
        #判断是否结束
        if terminate_prob > 0.6:
            self.isDone = 1
            terminal_reward = 50 + math.exp(-2*self.CurTime/self.TIME_Max)
            self.CurTotalReward += terminal_reward
            self.TotalReward += self.CurTotalReward
            print("--------------------------------------------------------目标达成！---------------------------------------------------------------------")
            str_ = f"第{self.episode}局结束，时间: {self.CurTime:.1f}s，奖励：{self.TotalReward:.6f}，terminate_prob = {terminate_prob:.6f}，" \
                   f"距离：{self.CurDistance:.2f}m，高度：{cur_self_info['Altitude']:.2f}m，攻击角：{self.CurAttackAngle*180/math.pi:.2f}，逃逸角：{self.CurEscapeAngle*180/math.pi:.2f}"
            print(str_)

        elif self.CurDistance > self.Distance_Max:
            self.isDone = 2
            terminal_reward = -50 + math.tanh(-math.fabs(self.CurAttackAngle) / math.pi)
            #terminal_reward = -50
            self.CurTotalReward += terminal_reward
            self.TotalReward += self.CurTotalReward
            str_ = f"第{self.episode}局结束，***距离***越界，时间：{self.CurTime:.1f}s，奖励：{self.TotalReward:.6f}，terminate_prob = {terminate_prob:.6f}，" \
                   f"距离：{self.CurDistance:.2f}m，高度：{cur_self_info['Altitude']:.2f}m，攻击角：{self.CurAttackAngle * 180 / math.pi:.2f}，逃逸角：{self.CurEscapeAngle * 180 / math.pi:.2f}"
            print(str_)

        elif cur_self_info['Altitude'] > self.ALT_Max or cur_self_info['Altitude'] < self.ALT_Min:
            self.isDone = 2
            #terminal_reward = -40 + math.tanh((self.WEZ_Min - self.CurDistance) / self.Distance_Max)
            terminal_reward = -50 + math.tanh(-math.fabs(self.CurAttackAngle) / math.pi)
            #terminal_reward = -50
            self.CurTotalReward += terminal_reward
            self.TotalReward += self.CurTotalReward
            str_ = f"第{self.episode}局结束，+++高度+++越界，时间：{self.CurTime:.1f}s，奖励：{self.TotalReward:.6f}，terminate_prob = {terminate_prob:.6f}，" \
                   f"距离：{self.CurDistance:.2f}m，高度：{cur_self_info['Altitude']:.2f}m，攻击角：{self.CurAttackAngle * 180 / math.pi:.2f}，逃逸角：{self.CurEscapeAngle * 180 / math.pi:.2f}"
            print(str_)

        elif self.CurTime >= self.TIME_Max:
            self.isDone = 3
            #terminal_reward = -30 + math.tanh((self.WEZ_Min - self.CurDistance) / self.Distance_Max)
            terminal_reward = -40 + math.tanh(-math.fabs(self.CurAttackAngle) / math.pi) + \
                              math.tanh((self.WEZ_Min - self.CurDistance) / self.Distance_Max)
            #terminal_reward = -30
            self.CurTotalReward += terminal_reward
            self.TotalReward += self.CurTotalReward
            str_ = f"第{self.episode}局结束，---时间---越界，时间：{self.CurTime:.1f}s，奖励：{self.TotalReward:.6f}，terminate_prob = {terminate_prob:.6f}，" \
                   f"距离：{self.CurDistance:.2f}m，高度：{cur_self_info['Altitude']:.2f}m，攻击角：{self.CurAttackAngle * 180 / math.pi:.2f}，逃逸角：{self.CurEscapeAngle * 180 / math.pi:.2f}"
            print(str_)

        if self.isDone != 0:
            self.TotalReward = 0
            self.episode += 1
        else:
            self.TotalReward += self.CurTotalReward


    # 计算奖励
    def getReward(self, self_info, dete_info, flag=False):
        distance, self_attackAngle, target_escapeAngle = self.calDisAttakEscape(self_info, dete_info)
        self_escapeAngle = math.pi - self_attackAngle
        target_attackAngle = math.pi - target_escapeAngle

        #[-1,1]
        potential_reward = (1 - (math.fabs(self_attackAngle) / math.pi + math.fabs(target_escapeAngle) / math.pi)) * \
                            math.exp(-0.0005 * math.fabs(distance - self.WEZ_Min))

        # # sparse reward function
        # sparse_reward = 0
        # # 敌机在我机范围内
        # if math.fabs(attackAngle) <= math.pi / 3 and math.fabs(escapeAngle) <= math.pi / 3 and \
        #         (distance >= self.WEZ_Min and distance <= self.WEZ_Max):
        #     sparse_reward = 1
        # # 我机在敌机范围内
        # if math.fabs(attackAngle) >= 2 * math.pi / 3 and math.fabs(escapeAngle) >= 2 * math.pi / 3 and \
        #         (distance >= self.WEZ_Min and distance <= self.WEZ_Max):
        #     sparse_reward = -1
        #
        # # potential function
        # potential_reward = (1 - (math.fabs(attackAngle) / math.pi + math.fabs(escapeAngle) / math.pi)) * \
        #                    math.exp(-0.0002 * math.fabs(distance - self.WEZ_Min))

        if flag is True:
            self.CurDistance = distance
            self.CurAttackAngle = self_attackAngle
            self.CurEscapeAngle = target_escapeAngle

        return potential_reward


    # 计算距离、攻击角、逃逸角
    def calDisAttakEscape(self, self_info, dete_info):
        # 计算两机距离
        distance = RongAoUtils.getDistance3D(dete_info)
        # 计算攻击角[0-pi]、逃逸角[0-pi]
        distanceVector3D = RongAoUtils.getDistanceVector3D(dete_info)
        selfPlaneSpeedVector = RongAoUtils.getSpeedVector3D(self_info)
        detePlaneSpeedVector = RongAoUtils.getSpeedVector3D(dete_info)
        self_attackAngle = math.acos(
            (selfPlaneSpeedVector[0] * distanceVector3D[0] + selfPlaneSpeedVector[1] * distanceVector3D[1] +
             selfPlaneSpeedVector[2] * distanceVector3D[2]) / distance)
        target_escapeAngle = math.acos(
            (detePlaneSpeedVector[0] * distanceVector3D[0] + detePlaneSpeedVector[1] * distanceVector3D[1] +
             detePlaneSpeedVector[2] * distanceVector3D[2]) / distance)

        return distance, self_attackAngle, target_escapeAngle


    # 获得一帧的状态
    def getOneFrameState(self, self_track, detectedInfo):
        # 提取1帧观测数据
        # 经纬度差[-180,180]、[-90,90]
        lon_diff = math.fabs(detectedInfo['Longitude'] - self_track['Longitude'])
        if lon_diff > 180:
            lon_diff = 360 - lon_diff
        lon_diff = lon_diff / 180.0
        lat_diff = math.fabs(detectedInfo['Latitude'] - self_track['Latitude']) / 90.0
        # 高度
        self_alt = self_track['Altitude'] / self.ALT_Max
        dete_alt = detectedInfo['Altitude'] / self.ALT_Max
        #alt_diff = (self_track['Altitude'] - detectedInfo['Altitude']) / self.ALT_Max
        # 速度差[0, 600]
        self_v_real = RongAoUtils.getSpeed(self_track)
        dete_v_real = RongAoUtils.getSpeed(detectedInfo)
        v_diff = (self_v_real - dete_v_real) / 600.0
        # 速度分量
        self_v_n = self_track['V_N'] / 600.0
        self_v_e = self_track['V_E'] / 600.0
        self_v_d = self_track['V_D'] / 600.0
        # 偏航[0, 2*pi]、俯仰[-pi/2, pi/2]、滚转[-pi, pi]
        self_heading = self_track['Heading'] / (2 * math.pi)
        self_pitch = self_track['Pitch'] / (math.pi / 2)
        self_roll = self_track['Roll'] / math.pi
        dete_heading = detectedInfo['Heading'] / (2 * math.pi)
        dete_pitch = detectedInfo['Pitch'] / (math.pi / 2)
        dete_roll = detectedInfo['Roll'] / math.pi
        # 状态向量
        one_state = np.array([lon_diff, lat_diff, self_alt, dete_alt, v_diff, self_v_n, self_v_e, self_v_d,
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


    def resetEpisode(self):
        print("第", self.episode, "局结束，本局被中断！！！  时间：", self.CurTime, ", TotalReward=", self.TotalReward)
        self.TotalReward = 0
        self.episode += 1




