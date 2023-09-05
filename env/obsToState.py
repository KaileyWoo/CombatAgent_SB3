import math
import numpy as np
from utils.RongAoUtils import RongAoUtils
from models.params import StateDim


class ObsToState:
    def __init__(self, env_id=0, port=8868):
        self.env_id = env_id+1
        self.port = port
        self.isDone = 0  #�Ƿ����ս��������ԭ��[0-δ���� 1-����ʤ 2-�з�ʤ 3-ʱ�䵽]
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
        self.episode = 0


    def proTerminal(self, result, role):
        self.episode += 1
        if result == 1:
            terminalStr = "�췽 ��ʤ��"
        elif result == 2:
            terminalStr = "���� ��ʤ��"
        else:
            terminalStr = "�Ծ�ʱ�䵽��"
        terminalStr = f"����{self.env_id}�顿��port={self.port}������ {self.episode} �ֽ�����****** {terminalStr} ******"
        if role == "red" and result == 1 or role == "blue" and result == 2:
            tReward = 50
            print(terminalStr + f"�� ʱ��: {self.CurTime:.1f}s���ܽ�����{self.TotalReward+tReward:.6f} ��--------------------------------------Ŀ���ɣ�")
        else:
            tReward = -50
            print(terminalStr + f"�� ʱ��: {self.CurTime:.1f}s���ܽ�����{self.TotalReward+tReward:.6f} ��")
        reward = tReward + self.TotalReward
        self.TotalReward = 0
        self.CurTotalReward = 0
        return reward


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
        terminal_reward = (1 - (math.fabs(self.CurAttackAngle) / math.pi + math.fabs(self.CurEscapeAngle) / math.pi)) * \
                          math.exp(-0.0005 * math.fabs(self.CurDistance - self.WEZ_Min))
        test = False
        if self.CurDistance > self.Distance_Max:
            self.isDone = 2
            if test:
                Reward = -50 + math.tanh(-math.fabs(self.CurAttackAngle) / math.pi)
                self.CurTotalReward = Reward
                self.TotalReward += self.CurTotalReward
                str_ = f"����{self.env_id}�顿��port={self.port}���� {self.episode} �ֽ�����***����***Խ�磬ʱ�䣺{self.CurTime:.1f}s���ܽ�����{self.TotalReward:.6f}�����룺{self.CurDistance:.2f}m��" \
                       f"�߶ȣ�{self_track['Altitude']:.2f}m�������ǣ�{self.CurAttackAngle * 180 / math.pi:.2f}�����ݽǣ�{self.CurEscapeAngle * 180 / math.pi:.2f}"
                print(str_)
        elif self_track['Altitude'] > self.ALT_Max or self_track['Altitude'] < self.ALT_Min:
            self.isDone = 2
            if test:
                Reward = -50 + math.tanh((self.WEZ_Max - self.CurDistance) / self.Distance_Max)
                self.CurTotalReward = Reward
                self.TotalReward += self.CurTotalReward
                str_ = f"����{self.env_id}�顿��port={self.port}���� {self.episode} �ֽ�����+++�߶�+++Խ�磬ʱ�䣺{self.CurTime:.1f}s���ܽ�����{self.TotalReward:.6f}�����룺{self.CurDistance:.2f}m��" \
                       f"�߶ȣ�{self_track['Altitude']:.2f}m�������ǣ�{self.CurAttackAngle * 180 / math.pi:.2f}�����ݽǣ�{self.CurEscapeAngle * 180 / math.pi:.2f}"
                print(str_)
        elif self.CurTime >= self.TIME_Max:
            self.isDone = 3
            if test:
                Reward = -50 + 0.8 * math.tanh((self.WEZ_Max - self.CurDistance) / self.Distance_Max) + 0.2 * math.tanh(
                    -math.fabs(self.CurAttackAngle) / math.pi)
                self.CurTotalReward = Reward
                self.TotalReward += self.CurTotalReward
                self.isDone = 3
                str_ = f"����{self.env_id}�顿��port={self.port}���� {self.episode} �ֽ�����---ʱ��---Խ�磬ʱ�䣺{self.CurTime:.1f}s���ܽ�����{self.TotalReward:.6f}�����룺{self.CurDistance:.2f}m��" \
                       f"�߶ȣ�{self_track['Altitude']:.2f}m�������ǣ�{self.CurAttackAngle * 180 / math.pi:.2f}�����ݽǣ�{self.CurEscapeAngle * 180 / math.pi:.2f}"
                print(str_)
        # elif math.fabs(self.CurAttackAngle) < math.pi / 3 and self.CurDistance < self.WEZ_Max:
        elif terminal_reward > 0.6 and self.CurDistance < self.WEZ_Max:
            self.isDone = 0
            if test:
                Reward = 50 + 0.5 * math.exp(-self.CurTime / self.TIME_Max) + 0.5 * terminal_reward
                self.CurTotalReward = Reward
                self.TotalReward += self.CurTotalReward
                print(
                    "--------------------------------------------------------Ŀ���ɣ�---------------------------------------------------------------------")
                str_ = f"����{self.env_id}�顿��port={self.port}���� {self.episode} �ֽ�����ʱ��: {self.CurTime:.1f}s���ܽ�����{self.TotalReward:.6f}���վֽ�����{terminal_reward:.6f}��" \
                       f"���룺{self.CurDistance:.2f}m���߶ȣ�{self_track['Altitude']:.2f}m�������ǣ�{self.CurAttackAngle * 180 / math.pi:.2f}�����ݽǣ�{self.CurEscapeAngle * 180 / math.pi:.2f}"
                print(str_)

        if test:
            if self.isDone != 0:
                self.TotalReward = 0
                self.episode += 1
            else:
                self.TotalReward += self.CurTotalReward
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




