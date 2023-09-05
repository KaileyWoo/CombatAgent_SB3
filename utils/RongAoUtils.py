import math

# ???????¦Ã??
CENTER_LON = 94.5
CENTER_LAT = 23.5
DBL_EPSILON = 2.2204460492503131e-016
R = 6371000  # ?????
NX = 1  # ???????????
NY = 3  # ???????????
ROLLING = 6  # ?????????
NROLLING = 5


class RongAoUtils:
    # ????????
    @staticmethod
    def degree2Rad(degree):
        return degree * math.pi / 180

    # ????????
    @staticmethod
    def radToDegree(rad):
        return rad * 180.0 / math.pi

    # ??¦Ã?xy
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

    # xy???¦Ã
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


    # ????????????-1??1?????ŽN???????
    @staticmethod
    def moveRL(action, id, x_pos, y_pos, r_pos, z_pos):
        """
        ????????????-1??1
        :param action: ???????
        :param x_pos: ?????¦Ë??
        :param y_pos: ?????¦Ë??
        :param r_pos: ????¦Ë??
        :param z_pos: ???¦Ë??
        :return:???????
        """
        action["msg_info"] = ["??????," + str(id) +",2,0,Delay,Force,0|" + str(x_pos) + "`" + str(y_pos) + "`" + str(r_pos) + "`" + str(z_pos)]

        return action


    # ??????????????????nx??ny??rolling
    @staticmethod
    def move(action, id, nx, ny, rolling):
        """
        :param action: ???????
        :param id: ??????????
        :param nx: ???????(???????)
        :param ny: ???????(???????????)
        :param rolling: (????????)
        :return:???????
        """
        action["msg_info"] = "PDI," + str(id) +",2,0,Delay,Force,0|" + str(nx) + "`" + str(ny) + "`" + str(rolling)
        return action

    # ??????
    @staticmethod
    def hitTarget(action, myPlaneID, enemyPlaneID):
        """
        :param action: ???????
        :param myPlaneID: ??????ID
        :param enemyPlaneID: ?§Ù????ID
        :return:???????
        """
        action[32]["map_order"][myPlaneID] = "????,15001,2,0,Delay,Force,0|5002"
        return action

    # ????§Ù??????????????????????????????????
    @staticmethod
    def getDistance3D(planeBlue):
        dX = planeBlue['Relative_X']
        dY = planeBlue['Relative_Y']
        dZ = planeBlue['Relative_Z']
        return math.sqrt(dX * dX + dY * dY + dZ * dZ)

    # ?????????????????????
    @staticmethod
    def getDistance2D(planeRed, planeBlue):
        dX = planeBlue['X'] - planeRed['X']
        dY = planeBlue['Y'] - planeRed['Y']
        return math.sqrt(dX * dX + dY * dY)

    # ????§Ù?????????????????????????????????????
    @staticmethod
    def getDistanceVector3D(planeBlue):
        dX = planeBlue['Relative_X']
        dY = planeBlue['Relative_Y']
        dZ = planeBlue['Relative_Z']
        return [dX, dY, dZ]

    # ???????????????????????
    @staticmethod
    def getSpeedVector3D(planeInfo):
        """
        ????????????????
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

    # ??????????????????????
    @staticmethod
    def getSpeed(planeInfo):
        """
        ????????????????
        """
        Speed = math.sqrt(planeInfo['V_D'] * planeInfo['V_D'] + planeInfo['V_E'] * planeInfo['V_E'] + planeInfo['V_N'] * planeInfo['V_N'])
        return Speed

# xy = RongAoUtils.ConvertLatLong_to_XY(94.5, 23.5)
# print('???????', xy)
# xy2 = RongAoUtils.ConvertLatLong_to_XY(94.4, 23.5)
# print('???????', xy2)
# y = RongAoUtils.ConvertXY_to_LatLong(xy[0], xy[1])
# print('????¦Ã', y)
