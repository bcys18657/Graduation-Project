# coding=utf-8
# ----------------使用仿真跟踪指定轨迹（存储于target.csv中）
# ----------------使用PD控制速度输入---------------------------
from __future__ import print_function

import os
import numpy as np
import csv
from dronekit import *
import pandas as pd
import FlightController
import speedControl

# connection_string = '/dev/ttyAMA0'
connection_string = '127.0.0.1:14550'
# connection_string = 'COM6'
baud_rate = 921600
global right, forward


class Kalman_Filter:
    def __init__(self):
        # 初始化
        # 状态量为位置和速度：x = [px, py, vx, vy]
        # 输入量为速度 ：[vx, vy]
        # 状态转移矩阵为 px(k+1) = px(k + 1) + vx * dt
        self.is_init = False
        # self.x_ = np.array([[self.px_, self.py_, self.vx_, self.vy_]]).T
        # # 传感器观测量
        # self.z_ = np.array([[self.px_, self.py_, self.vx_, self.vy_]]).T

        self.A = np.zeros([4, 4])
        self.B = np.zeros([4, 2])
        # 协方差矩阵，假设测量精度较好
        self.P = np.array([[5, 0, 0, 0],
                           [0, 5, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        # 协方差更新中的噪声矩阵 P = APA'+V
        self.V = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0 ,1]])

        # 传感器的观测值是位置，即z=[px, py]
        # 输出方程z = Hx + w
        # H = [1 0 0 0],
        #     [0 1 0 0]
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        # 传感器噪声
        self.W = np.array([[1, 0],
                           [0, 1]])

    def Init_x(self, px, py, vx, vy):
        # 初始化状态变量,之后只靠传感器输入z
        self.x_ = np.array([[px, py, vx, vy]]).T
        self.is_init = True

    def Set_A(self, dt):
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        self.B = np.array([[0, 0],
                           [0, 0],
                           [1, 0],
                           [0, 1]])

    def Get_Pos(self):
        return self.x_[0][0], self.x_[1][0]

    def IsInitilazed(self):
        return self.is_init

    def Prediction(self, u):
        # 预测步骤
        # u = np.array([[self.vx_, self.vy_]]).T
        self.x_ = np.dot(self.A, self.x_) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.V

    def Update(self, z):
        # 计算新息y = z - H' * x_pre
        y = z - np.dot(self.H, self.x_)

        # 计算卡尔曼增益
        S = np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.W)
        K = np.dot(np.dot(self.P, self.H.T), S)

        # 计算出最终结果
        self.x_ = self.x_ + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)


def get_target_traj():
    with open('target1.csv') as csvfile:
        reader = csv.reader(csvfile)
        return np.array([[float(row[1]), float(row[2])] for row in reader])


def UAV_init(port, height):
    # ***************UAV初始化**********************
    myCopter = FlightController.Vehicle(FCAddress=port)
    if not myCopter.initialize(simulation=False):
        return None
    # Try arming the vehicle
    timeoutCounter = 0
    while not myCopter.arm():
        timeoutCounter += 1
        if timeoutCounter > 3:
            print("Cannot arm the vehicle after 3 retries.")
            return None
    before_yaw = myCopter.vehicle.attitude.yaw

    if not myCopter.takeoff(height):
        return None
    after_yaw = myCopter.vehicle.attitude.yaw
    myCopter.goto(0.01, 0, 0)
    myCopter.condition_yaw(math.degrees(abs(after_yaw - before_yaw)),
                           clock_wise=True if after_yaw - before_yaw > 0 else False)
    return myCopter


def UAV_start_task(vehicle, traj):
    right = forward = 0
    last_right = last_forward = 0
    i = 0
    init = [-2.5, -2]
    dex = dey = 0
    ddex = ddey = 0
    log_file = pd.DataFrame([])
    start_time = time.time()
    last_time = time.time()
    kf = Kalman_Filter()

    while not vehicle.fsController.triggered and i < len(traj):
        if not kf.IsInitilazed():
            current_location = vehicle.vehicle.location.local_frame
            pos_x = current_location.east
            pos_y = current_location.north
            vx = vehicle.vehicle.velocity[0]
            vy = vehicle.vehicle.velocity[1]
            kf.Init_x(pos_x, pos_y, vx, vy)
            continue
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        kf.Set_A(dt)
        # print(dt)
        vx = vehicle.vehicle.velocity[0]
        vy = vehicle.vehicle.velocity[1]
        u = np.array([[vx, vy]]).T
        kf.Prediction(u)
        current_location = vehicle.vehicle.location.local_frame
        pos_x = current_location.east
        pos_y = current_location.north
        z = np.array([[pos_x, pos_y]]).T
        kf.Update(z)
        opt_x, opt_y = kf.Get_Pos()
        # if (current_time - start_time) / 0.1
        [target_x, target_y] = traj[i] + init

        # off_x = -pos_x + target_x
        # off_y = -pos_y + target_y
        off_x = -opt_x + target_x
        off_y = -opt_y + target_y
        # off = np.append(off, [off_x, off_y])

        addright, addforward, dex, dey, ddex, ddey = speedControl.speedControl(off_x, off_y, dex, dey, ddex, ddey)
        right += addright
        forward += addforward
        vehicle.send_nav_velocity(forward, right, 0)
        i = i + 1

        value = [[current_time - start_time,
                  # pos_x,
                  # pos_y,
                  opt_x,
                  opt_y,
                  off_x,
                  off_y,
                  target_x,
                  target_y,
                  vehicle.vehicle.velocity[0],
                  vehicle.vehicle.velocity[1]
                  ]]
        frame = pd.DataFrame(value)
        log_file = log_file.append(frame)
        log_file.to_csv('UAV_offset.csv', index=None)
        time.sleep(0.1)

    # off = off.reshape(-1, 2)
    # name = 'UAV_offset.csv'
    # file = pd.DataFrame(off)
    # file.to_csv(name)


if __name__ == "__main__":

    traj = get_target_traj()
    vehicle = UAV_init(connection_string, 3)
    if not vehicle:
        print("Progress exiting.")
        os._exit(1)
    # vehicle = UAV.connect_UAV(connection_string)
    time.sleep(5)
    UAV_start_task(vehicle, traj)
    vehicle.land()
    vehicle.exit()
