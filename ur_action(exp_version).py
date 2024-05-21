import pyrealsense2 as rs
import numpy as np
import cv2
import time
import math
import urx
from Dependencies.urx_custom.robotiq_two_finger_gripper import (
    Robotiq_Two_Finger_Gripper,
)


class realsense:
    def __init__(self, width=640, height=480, fps=30):

        print("Initializing Intel RealSense Camera")

        self.pipe = rs.pipeline()  # get realsense
        config = rs.config()  # settings
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipe)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.pipe.start()

        # profile = self.pipe.start()
        # depth_sensor = profile.get_device().first_depth_sensor()
        # depth_scale = depth_sensor.get_depth_scale()

        self.align = rs.align(rs.stream.color)

    def convert_pixel_to_3d_world(self, depth, x, y):
        upixel = np.array([float(x), float(y)], dtype=np.float32)
        distance = depth.get_distance(x, y)
        intrinsics = depth.get_profile().as_video_stream_profile().get_intrinsics()
        pcd = rs.rs2_deproject_pixel_to_point(intrinsics, upixel, distance)
        return pcd[2], -pcd[0], -pcd[1]


# [0,0,0] is the camera
# x - right
# y - down
# z - forward
# left up is (0,0), right is x , down is y


class robot:
    def __init__(self):
        self.robot = urx.Robot("192.168.1.117")
        self.robot.set_tcp((0, 0, 0, 0, 0, 0))
        # self.robot.set_payload(0.5, (0,0,0))
        time.sleep(0.2)
        self.gripper = Robotiq_Two_Finger_Gripper(self.robot)
        time.sleep(2)

        self.home_pose = []
        # the pose to see all parts of the table
        self.camera_pose = [
            3.1136746406555176,
            -2.5506375471698206,
            1.444279670715332,
            5.177735328674316,
            -1.6317957083331507,
            4.59913969039917,
        ]
        # Ready to manipulate
        self.ready_pose = [
            3.1140706539154053,
            -2.6063817183123987,
            2.1799821853637695,
            4.584079265594482,
            -1.6395896116839808,
            4.597281455993652,
        ]

        self.gripper.open_gripper()
        self.is_finger_opened = True

    def get_pose(self):
        return self.robot.getl()

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
        self.robot.movej(self.home_pose, acc=0.5, vel=0.5, relative=False)
        time.sleep(3)
        return

    def to_ready_pose(self):
        self.robot.movej(self.ready_pose, acc=0.5, vel=0.5, relative=False)
        time.sleep(3)
        return

    def to_camera_pose(self):
        self.robot.movej(self.camera_pose, acc=0.5, vel=0.5, relative=False)
        time.sleep(3)
        return

    # move end effector to (x,y,z) based on the base of the robot
    def move_end_effector(self, action):
        delta_pos = action[:3]

        # change xyz position
        pose = self.get_pose()

        pose[0] = delta_pos[0]
        pose[1] = delta_pos[1]
        pose[2] = delta_pos[2]

        self.robot.movel(pose, acc=0.1, vel=0.1)

        # t = self.robot.get_pose()
        # t.pos[0] += delta_pos[0]
        # t.pos[1] += delta_pos[1]
        # t.pos[2] += delta_pos[2]
        # t.orient.rotate_xb(math.radians(delta_ori[0]))
        # t.orient.rotate_yb(math.radians(delta_ori[1]))
        # t.orient.rotate_zb(math.radians(delta_ori[2]))
        # self.robot.set_pose(t, vel=1)

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

    # move [dx,dy,dz,drx,dry,drz,gripper]
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

        # get_orientation: rotation matrix ([lx, ly, lz])
        # each lx, ly, lz is a 3d vector based on the original coordinate system
        orientation = self.robot.get_orientation()[2]

        # state: xyz of gripper's end effector based on the arm's end effector
        state = orientation * arm_len

        R = self.get_rotation_matrix(r, p, y)

        # new_trans: how much gripper's xyz moved caused by rpy?
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

        if delta_grp == [1]:
            if self.is_finger_opened:
                self.finger_close()
            else:
                self.finger_open()
        return

    def trans_cam2base(self, x, y, z):
        coord_to_cam = np.asarray([x, y, z])
        trans = self.robot.get_orientation()
        tool_pose = self.robot.get_pos()

        cam_len = 0.075

        transnp = np.array([trans[2], -trans[1], trans[0]]).T

        # where is the camera based on the robot's end effector?: [0, 0, cam_len]
        del_cam = np.matmul(transnp, np.array([0, 0, cam_len]).reshape(-1, 1))

        cam_pose = np.array(
            [
                tool_pose[0] + del_cam[0],
                tool_pose[1] + del_cam[1],
                tool_pose[2] + del_cam[2],
            ]
        ).squeeze()

        x, y, z = np.matmul(transnp, coord_to_cam.reshape(-1, 1)).squeeze() + cam_pose

        print("target coord based on the base: {}, {}, {}".format(x, y, z))

        return x, y, z

    def pick(self, x, y, z):
        # x,y,z has a base on the camera. You MUST convert to the robot base
        self.to_camera_pose()

        x, y, z = self.trans_cam2base(x, y, z)

        if z < 0.03:
            z = 0.03
        self.finger_open()

        self.to_ready_pose()
        self.move_end_effector([x - 0.13, y + 0.01, z + 0.22])
        self.move_gripper([0.03, 0, -0.1, 0, 0, 0, 0])

        if y > 0:
            self.move_gripper([0.05, 0, 0, 0, 0, 0, 0])

        self.finger_close()
        self.move_gripper([0, 0, 0.2, 0, 0, 0, 0])

    def place(
        self, x, y, z
    ):  # x,y,z has a base on the camera. You MUST convert to the robot base
        self.to_camera_pose()

        x, y, z = self.trans_cam2base(x, y, z)

        if z < 0.03:
            z = 0.03

        self.to_ready_pose()
        self.move_end_effector([x - 0.06, y, z + 0.27])
        self.move_gripper([0, 0, -0.07, 0, 0, 0, 0])
        self.finger_open()

        return

    def open_cab(
        self, x, y, z
    ):  # x,y,z has a base on the camera. You MUST convert to the robot base
        self.to_camera_pose()

        x, y, z = self.trans_cam2base(x, y, z)

        if z < 0.03:
            z = 0.03
        self.finger_open()
        self.to_ready_pose()

        self.move_end_effector([x - 0.2, y + 0.02, 0.37])
        joint_pose = self.robot.getj()
        joint_pose[5] -= (math.pi) / 2
        self.robot.movej(joint_pose, acc=0.1, vel=0.3, relative=False)
        self.move_gripper([0.1, 0, 0, 0, -30, 0, 0])
        self.finger_close()
        self.move_gripper([0, 0.35, 0.2, 0, 0, 0, 0])
        self.finger_open()

    def close_cab(self, x, y, z):
        return

    def push(
        self, x, y, z
    ):  # x,y,z has a base on the camera. You MUST convert to the robot base
        self.to_camera_pose()

        x, y, z = self.trans_cam2base(x, y, z)

        if z < 0.03:
            z = 0.03
        self.finger_open()

        self.to_ready_pose()
        self.move_end_effector(
            [x - 0.12, y + 0.02, z + 0.13]
        )  # [x - 0.07, y + 0.03, z + 0.12]
        self.finger_close()
        self.move_gripper([0.15, 0, 0, 0, 0, 0, 0])

        self.move_gripper([-0.1, 0, 0.2, 0, 0, 0, 0])

    def shutdown_robot(self):
        self.robot.close()
        return


if __name__ == "__main__":

    save_image_path = "./realsense.png"
    point = (640, 390)
    camera = realsense(1280, 720, 30)
    arm = grasp()

    try:
        while True:
            arm.to_camera_pose()
            time.sleep(2)
            frames = camera.pipe.wait_for_frames()  #
            frames = camera.align.process(frames)
            aligned_depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            cv2.circle(color_image, (point[0], point[1]), 20, (0, 0, 255), 3)
            cv2.imwrite(save_image_path, color_image)

            tx, ty, tz = camera.convert_pixel_to_3d_world(
                aligned_depth_frame, point[0], point[1]
            )
            print("point based on the camera is ({},{},{})".format(tx, ty, tz))
            if tx == 0 and ty == 0 and tz == 0:
                continue

            arm.pick(x=tx, y=ty, z=tz)

            input("Press any key to continue")
    finally:
        camera.pipe.stop()
