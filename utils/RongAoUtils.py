import math

# ���ĵ�ľ�γ��
CENTER_LON = 94.5
CENTER_LAT = 23.5
DBL_EPSILON = 2.2204460492503131e-016
R = 6371000  # ����뾶
NX = 1  # ������طŴ���
NY = 3  # ������طŴ���
ROLLING = 6  # ��ת�ǷŴ���
NROLLING = 5


class RongAoUtils:
    # �Ƕ�ת����
    @staticmethod
    def degree2Rad(degree):
        return degree * math.pi / 180

    # ����ת�Ƕ�
    @staticmethod
    def radToDegree(rad):
        return rad * 180.0 / math.pi

    # ��γתxy
    @staticmethod
    def ConvertLatLong_to_XY(Lon, Lat):
        myLat = RongAoUtils.degree2Rad(Lat)
        myLon = RongAoUtils.degree2Rad(Lon)
        Center_Lat = RongAoUtils.degree2Rad(CENTER_LAT)
        Center_Lon = RongAoUtils.degree2Rad(CENTER_LON)

        delta_longitude = myLon - Center_Lon
        tmp = math.sin(myLat) * math.sin(Center_Lat) + math.cos(myLat) * math.cos(Center_Lat) * math.cos(
            delta_longitude)
        final_x = (R * math.cos(myLat) * math.sin(delta_longitude)) / tmp
        final_y = (R * (math.sin(myLat) * math.cos(Center_Lat) - math.cos(myLat) * math.sin(Center_Lat) * math.cos(
            delta_longitude))) / tmp

        return final_x, final_y

    # xyת��γ
    @staticmethod
    def ConvertXY_to_LatLong(x, y):
        my_x = x / R
        my_y = y / R
        Center_Lat = RongAoUtils.degree2Rad(CENTER_LAT)
        Center_Lon = RongAoUtils.degree2Rad(CENTER_LON)

        temp = math.sqrt(my_x * my_x + my_y * my_y)
        if -DBL_EPSILON < temp < DBL_EPSILON:
            Center_Lat = RongAoUtils.radToDegree(Center_Lat)
            Center_Lon = RongAoUtils.radToDegree(Center_Lon)
            return Center_Lat, Center_Lon

        c = math.atan(temp)

        final_lat = math.asin(math.cos(c) * math.sin(Center_Lat) + my_y * math.sin(c) * math.cos(Center_Lat) / temp)
        final_lon = Center_Lon + math.atan(
            my_x * math.sin(c) / (
                    temp * math.cos(Center_Lat) * math.cos(c) - my_y * math.sin(Center_Lat) * math.sin(c)))

        final_lat = RongAoUtils.radToDegree(final_lat)
        final_lon = RongAoUtils.radToDegree(final_lon)

        return final_lat, final_lon


    # ǿ��ѧϰ�Ķ�����-1��1�����嶯����ӳ��
    @staticmethod
    def moveRL(action, id, x_pos, y_pos, r_pos, z_pos):
        """
        ����Ķ�������-1��1
        :param action: �����ֵ�
        :param x_pos: �����λ��
        :param y_pos: �����λ��
        :param r_pos: ����λ��
        :param z_pos: �ŵ�λ��
        :return:�����ֵ�
        """
        action["msg_info"] = ["��ʻ�ٿ�," + str(id) +",2,0,Delay,Force,0|" + str(x_pos) + "`" + str(y_pos) + "`" + str(r_pos) + "`" + str(z_pos)]

        return action


    # ��ͨ������������ʵ��nx��ny��rolling
    @staticmethod
    def move(action, id, nx, ny, rolling):
        """
        :param action: �����ֵ�
        :param id: ���Ƶķɻ����
        :param nx: �������(�����ٶ�)
        :param ny: �������(���Ʒ�����ٶ�)
        :param rolling: (���ƹ�ת��)
        :return:�����ֵ�
        """
        action["msg_info"] = "PDI," + str(id) +",2,0,Delay,Force,0|" + str(nx) + "`" + str(ny) + "`" + str(rolling)
        return action

    # ���ָ��
    @staticmethod
    def hitTarget(action, myPlaneID, enemyPlaneID):
        """
        :param action: �����ֵ�
        :param myPlaneID: �ҷ��ɻ�ID
        :param enemyPlaneID: �з��ɻ�ID
        :return:�����ֵ�
        """
        action[32]["map_order"][myPlaneID] = "����,15001,2,0,Delay,Force,0|5002"
        return action

    # ͨ���з��ɻ��յ�����Ϣ���������ɻ�֮�����ά����
    @staticmethod
    def getDistance3D(planeBlue):
        dX = planeBlue['Relative_X']
        dY = planeBlue['Relative_Y']
        dZ = planeBlue['Relative_Z']
        return math.sqrt(dX * dX + dY * dY + dZ * dZ)

    # ���������ɻ�֮��Ķ�ά����
    @staticmethod
    def getDistance2D(planeRed, planeBlue):
        dX = planeBlue['X'] - planeRed['X']
        dY = planeBlue['Y'] - planeRed['Y']
        return math.sqrt(dX * dX + dY * dY)

    # ͨ���з��ɻ��յ�����Ϣ���������ɻ�֮�����ά����ʸ��
    @staticmethod
    def getDistanceVector3D(planeBlue):
        dX = planeBlue['Relative_X']
        dY = planeBlue['Relative_Y']
        dZ = planeBlue['Relative_Z']
        return [dX, dY, dZ]

    # ���ݷɻ���Ϣ����ɻ����ٶ�ʸ��
    @staticmethod
    def getSpeedVector3D(planeInfo):
        """
        ���ع�һ�����ٶ�ʸ��
        """
        # heading = math.pi/2 - planeInfo['Heading']
        # pitch = planeInfo['Pitch']
        # dx = math.cos(pitch) * math.cos(heading)
        # dy = math.cos(pitch) * math.sin(heading)
        # dz = math.sin(pitch)
        # dR = math.sqrt(dx * dx + dy * dy + dz * dz)
        # return [dx / dR, dy / dR, dz / dR]

        Speed = math.sqrt(planeInfo['V_D'] * planeInfo['V_D'] + planeInfo['V_E'] * planeInfo['V_E'] + planeInfo['V_N'] * planeInfo['V_N'])
        if Speed != 0:
            return [planeInfo['V_E'] / Speed, planeInfo['V_N'] / Speed, planeInfo['V_D'] / Speed]
        else:
            return [0, 0, 0]

    # ���ݷɻ���Ϣ����ɻ��ĶԵ��ٶ�
    @staticmethod
    def getSpeed(planeInfo):
        """
        ���ع�һ�����ٶ�ʸ��
        """
        Speed = math.sqrt(planeInfo['V_D'] * planeInfo['V_D'] + planeInfo['V_E'] * planeInfo['V_E'] + planeInfo['V_N'] * planeInfo['V_N'])
        return Speed

# xy = RongAoUtils.ConvertLatLong_to_XY(94.5, 23.5)
# print('ʵ������', xy)
# xy2 = RongAoUtils.ConvertLatLong_to_XY(94.4, 23.5)
# print('ʵ������', xy2)
# y = RongAoUtils.ConvertXY_to_LatLong(xy[0], xy[1])
# print('����γ', y)
