import cv2
import numpy as np
import time
import math
import urx
from pyk4a import PyK4A
from Dependencies.urx_custom.robotiq_two_finger_gripper import (
    Robotiq_Two_Finger_Gripper,
)
from math import pi


class RobotController:
    def __init__(self):
        self.arm = Arm()
        self.camera = PyK4A()
        self.camera.start()


class Arm:
    def __init__(self):
        self.robot = urx.Robot("192.168.1.117")
        self.robot.set_tcp((0, 0, 0, 0, 0, 0))
        # self.robot.set_payload(0.5, (0,0,0))
        time.sleep(0.2)
        self.gripper = Robotiq_Two_Finger_Gripper(self.robot)
        time.sleep(2)
        self.home_pose_down = [3.15, -1.98, 1.86, -1.43, 4.68, 1.64]
        self.home_pose_front = [
            2.1635491847991943,
            -1.9490583578692835,
            2.2103524208068848,
            -0.257486645375387,
            0.5646798610687256,
            1.564760446548462,
        ]  # [-0.011365238820211232, -1.2454717795001429, -2.188969437276022, -2.8360922972308558, 4.682430267333984, 1.5563137531280518]
        self.gripper.open_gripper()
        self.is_finger_opened = True

    def get_pose(self):
        pose = self.robot.getl()
        # print("current relative position: {}".format(pose))
        # print("current joint position: {}".format(self.robot.getj()))
        return pose

    def print_pose(self):
        print(self.robot.getj())
        return

    def finger_open(self):
        self.gripper.open_gripper()
        time.sleep(2)
        self.is_finger_opened = True
        return

    def finger_close(self):
        self.gripper.close_gripper()
        time.sleep(2)
        self.is_finger_opened = False
        return

    def to_home_pose(self):
        self.robot.movej(self.home_pose_front, acc=0.5, vel=0.5, relative=False)
        time.sleep(3)
        return

    def move_end_effector(self, action):
        delta_pos = action[:3]
        delta_ori = action[3:6]
        delta_grp = action[7:]

        if delta_grp != [0]:
            if self.is_finger_opened:
                self.finger_close()
            else:
                self.finger_open()

        # change xyz position
        pose = self.get_pose()

        # pose[0] += delta_pos[0]
        # pose[1] += delta_pos[1]
        # pose[2] += delta_pos[2]

        # self.robot.movel(pose, acc=1, vel=0.1)

        t = self.robot.get_pose()
        # print(t.pos)
        # print(type(t.pos))
        t.pos[0] += delta_pos[0]
        t.pos[1] += delta_pos[1]
        t.pos[2] += delta_pos[2]
        t.orient.rotate_xb(math.radians(delta_ori[0]))
        t.orient.rotate_yb(math.radians(delta_ori[1]))
        t.orient.rotate_zb(math.radians(delta_ori[2]))
        self.robot.set_pose(t, vel=1)

        time.sleep(2)

        return

    def get_rotation_matrix(self, roll, pitch, yaw):
        return np.asarray(
            [
                [
                    math.cos(pitch) * math.cos(yaw),
                    math.sin(roll) * math.sin(pitch) * math.cos(yaw)
                    - math.cos(roll) * math.sin(yaw),
                    math.cos(roll) * math.sin(pitch) * math.cos(yaw)
                    + math.sin(roll) * math.sin(yaw),
                ],
                [
                    math.cos(pitch) * math.sin(yaw),
                    math.sin(roll) * math.sin(pitch) * math.sin(yaw)
                    + math.cos(roll) * math.cos(yaw),
                    math.cos(roll) * math.sin(pitch) * math.sin(yaw)
                    - math.sin(roll) * math.cos(yaw),
                ],
                [
                    -math.sin(pitch),
                    math.sin(roll) * math.cos(pitch),
                    math.cos(roll) * math.cos(pitch),
                ],
            ],
            dtype=np.float32,
        )

    def move_gripper(self, action, arm_len=0.15):

        dist_xyz = action[:3]
        rpy = action[3:6]
        delta_grp = action[7:]

        (r, p, y) = rpy  # rpy in radian
        r = math.radians(r)
        p = math.radians(p)
        y = math.radians(y)

        trans = (
            self.robot.get_pose()
        )  # get current transformation matrix (tool to base)

        orientation = self.robot.get_orientation()[2]
        state = orientation * 0.15

        R = self.get_rotation_matrix(r, p, y)
        new_trans = -np.matmul(R, state.reshape(-1, 1)).squeeze()

        trans.pos.x += dist_xyz[0] + new_trans[0] + state[0]
        trans.pos.y += dist_xyz[1] + new_trans[1] + state[1]
        trans.pos.z += dist_xyz[2] + new_trans[2] + state[2]

        trans.orient.rotate_xb(r)
        trans.orient.rotate_yb(p)
        trans.orient.rotate_zb(y)

        self.robot.set_pose(trans, vel=0.1)
        time.sleep(2)

        if len(action) == 8:
            delta_grip_radian = math.radians(action[6])
            joint_pose = self.robot.getj()
            joint_pose[5] += delta_grip_radian
            self.robot.movej(joint_pose, acc=0.1, vel=0.3, relative=False)

        if delta_grp != [0]:
            if self.is_finger_opened:
                self.finger_close()
            else:
                self.finger_open()
        return

    def shutdown_robot(self):
        self.robot.close()
        return


if __name__ == "__main__":

    rc = RobotController()
    # rc.arm.to_home_pose()

    action = [0, 0, 0, 0, 0, 30, 0, 0]  # [x,y,z,rx,ry,rz,joint[5]]
    # rc.arm.move_end_effector(action)
    # rc.arm.move_gripper(action)
    time.sleep(2)

    rc.arm.get_pose()

    # rc.arm.to_home_pose()
    rc.arm.print_pose()
    rc.arm.shutdown_robot()
