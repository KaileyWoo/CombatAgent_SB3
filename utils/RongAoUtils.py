import math

# 中心点的经纬度
CENTER_LON = 94.5
CENTER_LAT = 23.5
DBL_EPSILON = 2.2204460492503131e-016
R = 6371000  # 地球半径
NX = 1  # 切向过载放大倍率
NY = 3  # 法向过载放大倍率
ROLLING = 6  # 滚转角放大倍率
NROLLING = 5


class RongAoUtils:
    # 角度转弧度
    @staticmethod
    def degree2Rad(degree):
        return degree * math.pi / 180

    # 弧度转角度
    @staticmethod
    def radToDegree(rad):
        return rad * 180.0 / math.pi

    # 经纬转xy
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

    # xy转经纬
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


    # 强化学习的动作，-1到1朝具体动作做映射
    @staticmethod
    def moveRL(action,id, x_pos, y_pos, r_pos, z_pos):
        """
        输入的动作都是-1到1
        :param action: 动作字典
        :param x_pos: 纵向杆位移
        :param y_pos: 横向杆位移
        :param r_pos: 油门位置
        :param z_pos: 脚蹬位置
        :return:动作字典
        """
        action["msg_info"] = "驾驶操控," + str(id) +",2,0,Delay,Force,0|" + str(x_pos) + "`" + str(y_pos) + "`" + str(r_pos) + "`" + str(z_pos)

        return action


    # 普通机动动作，真实的nx，ny，rolling
    @staticmethod
    def move(action, id, nx, ny, rolling):
        """
        :param action: 动作字典
        :param id: 控制的飞机编号
        :param nx: 切向过载(控制速度)
        :param ny: 法向过载(控制法向加速度)
        :param rolling: (控制滚转角)
        :return:动作字典
        """
        action["msg_info"] = "PDI," + str(id) +",2,0,Delay,Force,0|" + str(nx) + "`" + str(ny) + "`" + str(rolling)
        return action

    # 打击指令
    @staticmethod
    def hitTarget(action, myPlaneID, enemyPlaneID):
        """
        :param action: 动作字典
        :param myPlaneID: 我方飞机ID
        :param enemyPlaneID: 敌方飞机ID
        :return:动作字典
        """
        action[32]["map_order"][myPlaneID] = "发射,15001,2,0,Delay,Force,0|5002"
        return action

    # 通过敌方飞机收到的信息计算两个飞机之间的三维距离
    @staticmethod
    def getDistance3D(planeBlue):
        dX = planeBlue['Relative_X']
        dY = planeBlue['Relative_Y']
        dZ = planeBlue['Relative_Z']
        return math.sqrt(dX * dX + dY * dY + dZ * dZ)

    # 计算两个飞机之间的二维距离
    @staticmethod
    def getDistance2D(planeRed, planeBlue):
        dX = planeBlue['X'] - planeRed['X']
        dY = planeBlue['Y'] - planeRed['Y']
        return math.sqrt(dX * dX + dY * dY)

    # 通过敌方飞机收到的信息计算两个飞机之间的三维距离矢量
    @staticmethod
    def getDistanceVector3D(planeBlue):
        dX = planeBlue['Relative_X']
        dY = planeBlue['Relative_Y']
        dZ = planeBlue['Relative_Z']
        return [dX, dY, dZ]

    # 根据飞机信息计算飞机的速度矢量
    @staticmethod
    def getSpeedVector3D(planeInfo):
        """
        返回归一化的速度矢量
        """
        # heading = planeInfo['Heading']
        # pitch = planeInfo['Pitch']
        # dx = math.cos(pitch) * math.cos(heading)
        # dy = math.cos(pitch) * math.sin(heading)
        # dz = math.sin(pitch)
        # dR = math.sqrt(dx * dx + dy * dy + dz * dz)

        Speed = math.sqrt(planeInfo['V_D'] * planeInfo['V_D'] + planeInfo['V_E'] * planeInfo['V_E'] + planeInfo['V_N'] * planeInfo['V_N'])
        return [planeInfo['V_E'] / Speed, planeInfo['V_N'] / Speed, planeInfo['V_D'] / Speed]

    # 根据飞机信息计算飞机的对地速度
    @staticmethod
    def getSpeed(planeInfo):
        """
        返回归一化的速度矢量
        """
        Speed = math.sqrt(planeInfo['V_D'] * planeInfo['V_D'] + planeInfo['V_E'] * planeInfo['V_E'] + planeInfo['V_N'] * planeInfo['V_N'])
        return Speed

# xy = RongAoUtils.ConvertLatLong_to_XY(94.5, 23.5)
# print('实际坐标', xy)
# xy2 = RongAoUtils.ConvertLatLong_to_XY(94.4, 23.5)
# print('实际坐标', xy2)
# y = RongAoUtils.ConvertXY_to_LatLong(xy[0], xy[1])
# print('反求经纬', y)
