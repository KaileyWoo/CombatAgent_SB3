import math
import numpy as np
from utils.RongAoUtils import RongAoUtils
from models.params import StateDim


class ObsToState:
    def __init__(self):
        self.isDone = 0  #�Ƿ����ս��������ԭ��[0-δ���� 1-����ʤ 2-�з�ʤ[����ˤ��߶ȹ���] 3-ʱ�䵽]
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
        # ������1֡����
        self.PreSelfTrack, self.PreDetectedInfo = self.parseOneFrameData(first_info)
        # ������2֡����
        self.CurSelfTrack, self.CurDetectedInfo = self.parseOneFrameData(second_info)
        # ״̬����
        first_state = self.getStateVector(self.PreSelfTrack, self.PreDetectedInfo)
        second_stata = self.getStateVector(self.CurSelfTrack, self.CurDetectedInfo)
        STATE = np.concatenate((first_state, second_stata)).reshape(StateDim).astype(np.float32)
        return STATE


    def getStateRewardDoneForTwoFrames(self, first_info, second_info, time):
        """
        ���ݹ۲���Ϣ��2֡��������״̬�������������Ƿ����
        """
        self.isDone = 0

        STATE = self.getState(first_info, second_info, time)

        # ���㽱��
        self.CurTotalReward = self.getStepReward(self.PreSelfTrack, self.CurSelfTrack, self.PreDetectedInfo, self.CurDetectedInfo)
        self.is_done(self.CurSelfTrack)

        return STATE, self.CurTotalReward, self.isDone


    # �ж��Ƿ����
    def is_done(self, self_track):
        if self.CurDistance > self.Distance_Max:
            Reward = -50 + math.tanh(-math.fabs(self.CurAttackAngle) / math.pi)
            self.CurTotalReward = Reward
            self.TotalReward += self.CurTotalReward
            self.isDone = 2
            str_ = f"��{self.episode}�ֽ�����***����***Խ�磬ʱ�䣺{self.CurTime:.1f}s��������{self.TotalReward:.6f}�����룺{self.CurDistance:.2f}m��" \
                   f"�߶ȣ�{self_track['Altitude']:.2f}m�������ǣ�{self.CurAttackAngle * 180 / math.pi:.2f}�����ݽǣ�{self.CurEscapeAngle * 180 / math.pi:.2f}"
            print(str_)
        elif self_track['Altitude'] > self.ALT_Max or self_track['Altitude'] < self.ALT_Min:
            Reward = -50 + math.tanh((self.WEZ_Max - self.CurDistance) / self.Distance_Max)
            self.CurTotalReward = Reward
            self.TotalReward += self.CurTotalReward
            self.isDone = 2
            str_ = f"��{self.episode}�ֽ�����+++�߶�+++Խ�磬ʱ�䣺{self.CurTime:.1f}s��������{self.TotalReward:.6f}�����룺{self.CurDistance:.2f}m��" \
                   f"�߶ȣ�{self_track['Altitude']:.2f}m�������ǣ�{self.CurAttackAngle * 180 / math.pi:.2f}�����ݽǣ�{self.CurEscapeAngle * 180 / math.pi:.2f}"
            print(str_)
        elif self.CurTime >= self.TIME_Max:
            Reward = -50 + 0.8 * math.tanh((self.WEZ_Max - self.CurDistance) / self.Distance_Max) + 0.2 * math.tanh(-math.fabs(self.CurAttackAngle) / math.pi)
            self.CurTotalReward = Reward
            self.TotalReward += self.CurTotalReward
            self.isDone = 3
            str_ = f"��{self.episode}�ֽ�����---ʱ��---Խ�磬ʱ�䣺{self.CurTime:.1f}s��������{self.TotalReward:.6f}�����룺{self.CurDistance:.2f}m��" \
                   f"�߶ȣ�{self_track['Altitude']:.2f}m�������ǣ�{self.CurAttackAngle * 180 / math.pi:.2f}�����ݽǣ�{self.CurEscapeAngle * 180 / math.pi:.2f}"
            print(str_)
        elif math.fabs(self.CurAttackAngle) < math.pi / 3 and self.CurDistance < self.WEZ_Max:
            Reward = 50 + 0.8*math.exp(-self.CurTime / self.TIME_Max) + 0.2*math.exp(-(math.fabs(self.CurEscapeAngle))/math.pi)
            self.CurTotalReward = Reward
            self.TotalReward += self.CurTotalReward
            self.isDone = 1
            print("--------------------------------------------------------Ŀ���ɣ�---------------------------------------------------------------------")
            str_ = f"��{self.episode}�ֽ�����ʱ��: {self.CurTime:.1f}s��������{self.TotalReward:.6f}�����룺{self.CurDistance:.2f}m��" \
                   f"�߶ȣ�{self_track['Altitude']:.2f}m�������ǣ�{self.CurAttackAngle * 180 / math.pi:.2f}�����ݽǣ�{self.CurEscapeAngle * 180 / math.pi:.2f}"
            print(str_)

        if self.isDone != 0:
            self.TotalReward = 0
            self.episode += 1
        else:
            self.TotalReward += self.CurTotalReward


    # ���㵥������
    def getStepReward(self, first_self_track, second_self_track, first_DetectedInfo, second_DetectedInfo):

        pre_distance, pre_self_attackAngle, pre_target_escapeAngle = self.calDisAttakEscape(first_self_track, first_DetectedInfo)
        distance, self_attackAngle, target_escapeAngle = self.calDisAttakEscape(second_self_track, second_DetectedInfo)
        self.CurDistance = distance
        self.CurAttackAngle = self_attackAngle
        self.CurEscapeAngle = target_escapeAngle

        # ���뽱��
        pre_distance -= self.WEZ_Max
        distance -= self.WEZ_Max
        dis_reward = math.tanh((pre_distance - distance) / 10)
        # �����ǽ���
        attack_reward = math.tanh((pre_self_attackAngle - self_attackAngle) * 100)
        # ���ݽǽ���
        escape_reward = math.tanh((pre_target_escapeAngle - target_escapeAngle) * 100)

        Reward = 0.4 * dis_reward + 0.4 * attack_reward + 0.1 * escape_reward + 0.1 * math.tanh(-self.CurTime / self.TIME_Max)
        Reward = min(1, max(-1, Reward))
        Reward *= 0.01

        return Reward


    # ���״̬����
    def getStateVector(self, self_info, dete_info):
        distance, self_attackAngle, target_escapeAngle = self.calDisAttakEscape(self_info, dete_info)
        # �߶�
        self_alt = self_info['Altitude'] / self.ALT_Max
        dete_alt = dete_info['Altitude'] / self.ALT_Max
        # alt_diff = (self_track['Altitude'] - detectedInfo['Altitude']) / self.ALT_Max
        # �ٶȲ�[0, 600]
        self_v_real = RongAoUtils.getSpeed(self_info) / 600.0
        dete_v_real = RongAoUtils.getSpeed(dete_info) / 600.0
        #v_diff = (self_v_real - dete_v_real)
        # �ٶȷ���
        self_v_n = self_info['V_N'] / 600.0
        self_v_e = self_info['V_E'] / 600.0
        self_v_d = self_info['V_D'] / 600.0
        # ���ٶȷ���
        self_a_x = self_info['accelerations_x'] / 100.0
        self_a_y = self_info['accelerations_y'] / 100.0
        self_a_z = self_info['accelerations_z'] / 100.0
        # ƫ��[0, 2*pi]������[-pi/2, pi/2]����ת[-pi, pi]
        self_heading = self_info['Heading'] / (2 * math.pi)
        self_pitch = self_info['Pitch'] / (math.pi / 2)
        self_roll = self_info['Roll'] / math.pi
        dete_heading = dete_info['Heading'] / (2 * math.pi)
        dete_pitch = dete_info['Pitch'] / (math.pi / 2)
        dete_roll = dete_info['Roll'] / math.pi
        # ��������
        self_alpha = self_info['alpha'] / (2 * math.pi)    # ӭ��/����
        self_beta = self_info['beta'] / (2 * math.pi)      # �໬��/���ƽ�
        self_p = self_info['p'] / math.pi  # ��ת���ٶ�(����ÿ��)
        self_q = self_info['q'] / math.pi  # �������ٶ�(����ÿ��)
        self_r = self_info['r'] / math.pi  # �໬���ٶ�(����ÿ��)

        # ״̬����
        state = np.array([distance/self.Distance_Max, self_attackAngle/math.pi, target_escapeAngle/math.pi, self_alt, dete_alt,
                          self_v_real, dete_v_real, self_v_n, self_v_e, self_v_d, self_a_x, self_a_y, self_a_z,
                          self_heading, self_pitch, self_roll, dete_heading, dete_pitch, dete_roll,
                          self_alpha, self_beta, self_p, self_q, self_r],dtype=np.float32)

        return state


    # ���״̬����������־
    def getStateRewardDone(self, first_info, second_info, third_info, time):
        """
        ���ݹ۲���Ϣ��3֡��������״̬�������������Ƿ����
        """
        self.CurTime = time
        self.isDone = 0

        # ������1֡����
        pre_self_track, pre_DetectedInfo = self.parseOneFrameData(first_info)
        # ������2֡����
        cur_self_track, cur_DetectedInfo = self.parseOneFrameData(second_info)
        # ������3֡����
        next_self_track, next_DetectedInfo = self.parseOneFrameData(third_info)
        # ��ȡ��1֡�۲�����
        pre_state = self.getOneFrameState(pre_self_track, pre_DetectedInfo)
        # ��ȡ��2֡�۲�����
        cur_state = self.getOneFrameState(cur_self_track, cur_DetectedInfo)
        # ��ȡ��3֡�۲�����
        next_state = self.getOneFrameState(next_self_track, next_DetectedInfo)
        # ״̬����
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
        #�ж��Ƿ����
        if terminate_prob > 0.6:
            self.isDone = 1
            terminal_reward = 50 + math.exp(-2*self.CurTime/self.TIME_Max)
            self.CurTotalReward += terminal_reward
            self.TotalReward += self.CurTotalReward
            print("--------------------------------------------------------Ŀ���ɣ�---------------------------------------------------------------------")
            str_ = f"��{self.episode}�ֽ�����ʱ��: {self.CurTime:.1f}s��������{self.TotalReward:.6f}��terminate_prob = {terminate_prob:.6f}��" \
                   f"���룺{self.CurDistance:.2f}m���߶ȣ�{cur_self_info['Altitude']:.2f}m�������ǣ�{self.CurAttackAngle*180/math.pi:.2f}�����ݽǣ�{self.CurEscapeAngle*180/math.pi:.2f}"
            print(str_)

        elif self.CurDistance > self.Distance_Max:
            self.isDone = 2
            terminal_reward = -50 + math.tanh(-math.fabs(self.CurAttackAngle) / math.pi)
            #terminal_reward = -50
            self.CurTotalReward += terminal_reward
            self.TotalReward += self.CurTotalReward
            str_ = f"��{self.episode}�ֽ�����***����***Խ�磬ʱ�䣺{self.CurTime:.1f}s��������{self.TotalReward:.6f}��terminate_prob = {terminate_prob:.6f}��" \
                   f"���룺{self.CurDistance:.2f}m���߶ȣ�{cur_self_info['Altitude']:.2f}m�������ǣ�{self.CurAttackAngle * 180 / math.pi:.2f}�����ݽǣ�{self.CurEscapeAngle * 180 / math.pi:.2f}"
            print(str_)

        elif cur_self_info['Altitude'] > self.ALT_Max or cur_self_info['Altitude'] < self.ALT_Min:
            self.isDone = 2
            #terminal_reward = -40 + math.tanh((self.WEZ_Min - self.CurDistance) / self.Distance_Max)
            terminal_reward = -50 + math.tanh(-math.fabs(self.CurAttackAngle) / math.pi)
            #terminal_reward = -50
            self.CurTotalReward += terminal_reward
            self.TotalReward += self.CurTotalReward
            str_ = f"��{self.episode}�ֽ�����+++�߶�+++Խ�磬ʱ�䣺{self.CurTime:.1f}s��������{self.TotalReward:.6f}��terminate_prob = {terminate_prob:.6f}��" \
                   f"���룺{self.CurDistance:.2f}m���߶ȣ�{cur_self_info['Altitude']:.2f}m�������ǣ�{self.CurAttackAngle * 180 / math.pi:.2f}�����ݽǣ�{self.CurEscapeAngle * 180 / math.pi:.2f}"
            print(str_)

        elif self.CurTime >= self.TIME_Max:
            self.isDone = 3
            #terminal_reward = -30 + math.tanh((self.WEZ_Min - self.CurDistance) / self.Distance_Max)
            terminal_reward = -40 + math.tanh(-math.fabs(self.CurAttackAngle) / math.pi) + \
                              math.tanh((self.WEZ_Min - self.CurDistance) / self.Distance_Max)
            #terminal_reward = -30
            self.CurTotalReward += terminal_reward
            self.TotalReward += self.CurTotalReward
            str_ = f"��{self.episode}�ֽ�����---ʱ��---Խ�磬ʱ�䣺{self.CurTime:.1f}s��������{self.TotalReward:.6f}��terminate_prob = {terminate_prob:.6f}��" \
                   f"���룺{self.CurDistance:.2f}m���߶ȣ�{cur_self_info['Altitude']:.2f}m�������ǣ�{self.CurAttackAngle * 180 / math.pi:.2f}�����ݽǣ�{self.CurEscapeAngle * 180 / math.pi:.2f}"
            print(str_)

        if self.isDone != 0:
            self.TotalReward = 0
            self.episode += 1
        else:
            self.TotalReward += self.CurTotalReward


    # ���㽱��
    def getReward(self, self_info, dete_info, flag=False):
        distance, self_attackAngle, target_escapeAngle = self.calDisAttakEscape(self_info, dete_info)
        self_escapeAngle = math.pi - self_attackAngle
        target_attackAngle = math.pi - target_escapeAngle

        #[-1,1]
        potential_reward = (1 - (math.fabs(self_attackAngle) / math.pi + math.fabs(target_escapeAngle) / math.pi)) * \
                            math.exp(-0.0005 * math.fabs(distance - self.WEZ_Min))

        # # sparse reward function
        # sparse_reward = 0
        # # �л����һ���Χ��
        # if math.fabs(attackAngle) <= math.pi / 3 and math.fabs(escapeAngle) <= math.pi / 3 and \
        #         (distance >= self.WEZ_Min and distance <= self.WEZ_Max):
        #     sparse_reward = 1
        # # �һ��ڵл���Χ��
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


    # ������롢�����ǡ����ݽ�
    def calDisAttakEscape(self, self_info, dete_info):
        # ������������
        distance = RongAoUtils.getDistance3D(dete_info)
        # ���㹥����[0-pi]�����ݽ�[0-pi]
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


    # ���һ֡��״̬
    def getOneFrameState(self, self_track, detectedInfo):
        # ��ȡ1֡�۲�����
        # ��γ�Ȳ�[-180,180]��[-90,90]
        lon_diff = math.fabs(detectedInfo['Longitude'] - self_track['Longitude'])
        if lon_diff > 180:
            lon_diff = 360 - lon_diff
        lon_diff = lon_diff / 180.0
        lat_diff = math.fabs(detectedInfo['Latitude'] - self_track['Latitude']) / 90.0
        # �߶�
        self_alt = self_track['Altitude'] / self.ALT_Max
        dete_alt = detectedInfo['Altitude'] / self.ALT_Max
        #alt_diff = (self_track['Altitude'] - detectedInfo['Altitude']) / self.ALT_Max
        # �ٶȲ�[0, 600]
        self_v_real = RongAoUtils.getSpeed(self_track)
        dete_v_real = RongAoUtils.getSpeed(detectedInfo)
        v_diff = (self_v_real - dete_v_real) / 600.0
        # �ٶȷ���
        self_v_n = self_track['V_N'] / 600.0
        self_v_e = self_track['V_E'] / 600.0
        self_v_d = self_track['V_D'] / 600.0
        # ƫ��[0, 2*pi]������[-pi/2, pi/2]����ת[-pi, pi]
        self_heading = self_track['Heading'] / (2 * math.pi)
        self_pitch = self_track['Pitch'] / (math.pi / 2)
        self_roll = self_track['Roll'] / math.pi
        dete_heading = detectedInfo['Heading'] / (2 * math.pi)
        dete_pitch = detectedInfo['Pitch'] / (math.pi / 2)
        dete_roll = detectedInfo['Roll'] / math.pi
        # ״̬����
        one_state = np.array([lon_diff, lat_diff, self_alt, dete_alt, v_diff, self_v_n, self_v_e, self_v_d,
                              self_heading, self_pitch, self_roll, dete_heading, dete_pitch, dete_roll],
                             dtype=np.float32)

        return one_state


    # ����һ֡����
    def parseOneFrameData(self, observation):
        cur_DetectedInfo = {}
        cur_WeaponSystem = {}
        cur_self_track = {}
        cur_SFC = {}
        cur_MissileTrack = {}
        for data in observation:
            # ̽����Ϣ
            if data["data_tp"] == "DetectedInfo":
                if len(data["data_info"][0]['DetectedTargets']) > 0:
                    cur_DetectedInfo = data["data_info"][0]['DetectedTargets'][0]
            # ������Ϣ
            elif data["data_tp"] == "WeaponSystem":
                cur_WeaponSystem = data["data_info"][0]
            # ��������Ϣ
            elif data["data_tp"] == "track":
                cur_self_track = data["data_info"][0]
            # �����Ϣ
            elif data["data_tp"] == "SFC":
                cur_SFC = data["data_info"]
            # ������Ϣ
            elif data["data_tp"] == "MissileTrack":
                cur_MissileTrack = data["data_info"]

        return cur_self_track, cur_DetectedInfo


    def resetEpisode(self):
        print("��", self.episode, "�ֽ��������ֱ��жϣ�����  ʱ�䣺", self.CurTime, ", TotalReward=", self.TotalReward)
        self.TotalReward = 0
        self.episode += 1




